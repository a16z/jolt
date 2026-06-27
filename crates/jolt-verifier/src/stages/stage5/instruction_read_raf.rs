//! The stage 5 `InstructionReadRaf` sumcheck instance.
//!
//! The most intricate stage 5 relation: its output `Expr` references indexed
//! opening families (lookup-table flags, virtual RA chunks) and point-derived
//! *publics* (`EqTableValue`, `EqRafConstant`, `EqRafFlag`) computed from the
//! instruction address/cycle points and the upstream claim-reduction point. The
//! full instruction address is split across the virtual-RA opening points, so
//! `derive_output_term` reconstructs it from the output opening cells.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafChallenges, InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::instruction::InstructionReadRafDimensions, InstructionReadRafChallenge,
    InstructionReadRafPublic, JoltChallengeId, JoltDerivedId, JoltRelationId,
};
use jolt_claims::{SumcheckChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{
    try_eq_mle, IdentityPolynomial, MultilinearEvaluation, OperandPolynomial, OperandSide,
};

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage2::Stage2ClearOutput;
use crate::VerifierError;

/// Wire the consumed openings from the upstream instruction claim-reduction
/// (stage 2), applying the lookup-output fallback to the product remainder.
/// All three share the claim-reduction opening point. (Verifier-side constructor
/// for the moved [`InstructionReadRafInputClaims`].)
pub fn instruction_read_raf_inputs_from_upstream<F: Field>(
    stage2: &Stage2ClearOutput<F>,
) -> InstructionReadRafInputClaims<OpeningClaim<F>> {
    let reduction = &stage2.output_claims.instruction_claim_reduction;
    let lookup_output = reduction.lookup_output.as_ref().map_or(
        stage2.output_claims.product_remainder.lookup_output.value,
        |claim| claim.value,
    );
    let point = stage2
        .output_claims
        .instruction_claim_reduction_point()
        .to_vec();
    InstructionReadRafInputClaims {
        lookup_output: OpeningClaim {
            point: point.clone(),
            value: lookup_output,
        },
        left_lookup_operand: OpeningClaim {
            point: point.clone(),
            value: reduction.left_lookup_operand.value,
        },
        right_lookup_operand: OpeningClaim {
            point,
            value: reduction.right_lookup_operand.value,
        },
    }
}

pub struct InstructionReadRaf<F: Field> {
    symbolic: relations::instruction::ReadRaf,
    dimensions: InstructionReadRafDimensions,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> InstructionReadRaf<F> {
    pub fn new(dimensions: InstructionReadRafDimensions) -> Self {
        Self {
            symbolic: relations::instruction::ReadRaf::new(dimensions),
            dimensions,
            _field: core::marker::PhantomData,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: reason.to_string(),
    }
}

/// Reconstruct the instruction address point from the virtual-RA opening cells:
/// each RA opening point is `chunk ++ r_cycle`, and the chunks tile the address
/// in order, so stripping the trailing cycle and concatenating recovers it.
pub(crate) fn reconstruct_r_address<F: Field, C: GetPoint<F>>(
    outputs: &InstructionReadRafOutputClaims<C>,
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

impl<F: Field> ConcreteSumcheck<F> for InstructionReadRaf<F> {
    type Symbolic = relations::instruction::ReadRaf;
    type Inputs<C> = InstructionReadRafInputClaims<C>;
    type Outputs<C> = InstructionReadRafOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &InstructionReadRafInputClaims<C>,
    ) -> Result<InstructionReadRafOutputClaims<Vec<F>>, VerifierError> {
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
        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag: opening_point.r_cycle,
        })
    }

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        inputs: &InstructionReadRafInputClaims<C>,
        outputs: &InstructionReadRafOutputClaims<OpeningClaim<F>>,
        challenges: &InstructionReadRafChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::InstructionReadRaf(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
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
        // The RAF publics fold the batching gamma into the operand evaluations. The
        // gamma comes from the drawn `challenges` struct (the same value
        // `draw_challenges` produced), not a stored scalar.
        let gamma = challenges
            .resolve_challenge(&JoltChallengeId::from(InstructionReadRafChallenge::Gamma))
            .ok_or(VerifierError::MissingStageClaimChallenge {
                id: JoltChallengeId::from(InstructionReadRafChallenge::Gamma),
            })?;
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
