//! The stage 5 `InstructionReadRaf` sumcheck instance.
//!
//! The most intricate stage 5 relation: its output `Expr` references indexed
//! opening families (lookup-table flags, virtual RA chunks) and point-derived
//! *publics* (`EqTableValue`, `EqRafConstant`, `EqRafFlag`) computed from the
//! instruction address/cycle points and the upstream claim-reduction point. The
//! full instruction address is split across the virtual-RA opening points, so
//! `resolve_public` reconstructs it from the located output cells.

use jolt_claims::protocols::jolt::{
    formulas::instruction::{self, InstructionReadRafDimensions},
    InstructionReadRafChallenge, InstructionReadRafPublic, JoltChallengeId, JoltPublicId,
    JoltRelationClaims, JoltRelationId,
};
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{
    try_eq_mle, IdentityPolynomial, MultilinearEvaluation, OperandPolynomial, OperandSide,
};

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckInstance};
use crate::stages::stage5::inputs::{
    InstructionReadRafInputs, InstructionReadRafOutputOpeningClaims,
};
use crate::VerifierError;

pub struct InstructionReadRafRelation<F: Field> {
    claims: JoltRelationClaims<F>,
    dimensions: InstructionReadRafDimensions,
    gamma: F,
}

impl<F: Field> InstructionReadRafRelation<F> {
    pub fn new(dimensions: InstructionReadRafDimensions, gamma: F) -> Self {
        Self {
            claims: instruction::read_raf(dimensions),
            dimensions,
            gamma,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: reason.to_string(),
    }
}

/// Reconstruct the instruction address point from the located virtual-RA cells:
/// each RA opening point is `chunk ++ r_cycle`, and the chunks tile the address
/// in order, so stripping the trailing cycle and concatenating recovers it.
fn reconstruct_r_address<F: Field>(
    outputs: &InstructionReadRafOutputOpeningClaims<OpeningClaim<F>>,
    cycle_len: usize,
) -> Vec<F> {
    outputs
        .instruction_ra
        .iter()
        .flat_map(|cell| {
            let point = cell.point();
            point[..point.len() - cycle_len].iter().copied()
        })
        .collect()
}

impl<F: Field> SumcheckInstance<F> for InstructionReadRafRelation<F> {
    type Inputs<C> = InstructionReadRafInputs<C>;
    type Outputs<C> = InstructionReadRafOutputOpeningClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_output_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &InstructionReadRafInputs<C>,
    ) -> Result<InstructionReadRafOutputOpeningClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .dimensions
            .opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        let ra_count = self.dimensions.num_virtual_ra_polys();
        let chunk_size = opening_point
            .r_address
            .len()
            .checked_div(ra_count)
            .filter(|chunk_size| chunk_size * ra_count == opening_point.r_address.len())
            .ok_or_else(|| {
                public_input_failed(format!(
                    "instruction address point length {} is not divisible by virtual RA count {ra_count}",
                    opening_point.r_address.len()
                ))
            })?;
        let instruction_ra = opening_point
            .r_address
            .chunks(chunk_size)
            .map(|chunk| [chunk, opening_point.r_cycle.as_slice()].concat())
            .collect::<Vec<_>>();
        let lookup_table_flags =
            vec![opening_point.r_cycle.clone(); LookupTableKind::<RISCV_XLEN>::COUNT];
        Ok(InstructionReadRafOutputOpeningClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag: opening_point.r_cycle,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::Gamma) => {
                Ok(self.gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        inputs: &InstructionReadRafInputs<C>,
        outputs: &InstructionReadRafOutputOpeningClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::InstructionReadRaf(public) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        let r_cycle = outputs.instruction_raf_flag.point();
        let r_address = reconstruct_r_address(outputs, r_cycle.len());
        // eq over the upstream instruction claim-reduction cycle point; all three
        // consumed openings share that point, so the lookup-output input carries it.
        let eq_reduction =
            try_eq_mle(inputs.lookup_output.point(), r_cycle).map_err(public_input_failed)?;
        let address_bits = self.dimensions.instruction_address_bits();
        let left = || OperandPolynomial::new(address_bits, OperandSide::Left).evaluate(&r_address);
        let right =
            || OperandPolynomial::new(address_bits, OperandSide::Right).evaluate(&r_address);
        let gamma = self.gamma;
        let gamma2 = gamma * gamma;
        match public {
            InstructionReadRafPublic::EqTableValue(index) => {
                let table = LookupTableKind::<RISCV_XLEN>::iter()
                    .find(|table| table.index() == *index)
                    .ok_or_else(|| {
                        public_input_failed(format!("unknown lookup table index {index}"))
                    })?;
                Ok(eq_reduction * table.evaluate_mle::<F, F>(&r_address))
            }
            InstructionReadRafPublic::EqRafConstant => {
                Ok(eq_reduction * (gamma * left() + gamma2 * right()))
            }
            InstructionReadRafPublic::EqRafFlag => {
                let identity = IdentityPolynomial::new(address_bits).evaluate(&r_address);
                Ok(eq_reduction * (gamma2 * identity - gamma * left() - gamma2 * right()))
            }
        }
    }
}
