//! The stage 6 `InstructionRaVirtualization` cycle-phase sumcheck instance.
//!
//! Virtualizes the per-virtual reduced `InstructionRa` claims (from the stage-5
//! instruction read-RAF) into the per-chunk committed `InstructionRa` openings
//! that the stage-7 hamming-weight reduction consumes. Each produced opening
//! shares the cycle suffix derived from this sumcheck; the address prefix is the
//! stage-5 instruction address point. Its only public, `EqCycle`, ties the
//! produced cycle to the stage-5 instruction read-RAF cycle.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionRaVirtualizationInputClaims, InstructionRaVirtualizationOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::committed_address_chunks, instruction::InstructionRaVirtualizationDimensions,
    },
    InstructionRaVirtualizationChallenge, InstructionRaVirtualizationPublic, JoltChallengeId,
    JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage5::Stage5ClearOutput;
use crate::VerifierError;

/// Wire the per-virtual reduced `InstructionRa` openings from the stage-5
/// instruction read-RAF. (Verifier-side constructor for the moved
/// [`InstructionRaVirtualizationInputClaims`].)
pub fn instruction_ra_virtualization_inputs_from_upstream<F: Field>(
    stage5: &Stage5ClearOutput<F>,
) -> InstructionRaVirtualizationInputClaims<OpeningClaim<F>> {
    InstructionRaVirtualizationInputClaims {
        instruction_ra: stage5
            .output_claims
            .instruction_read_raf
            .instruction_ra
            .clone(),
    }
}

pub struct InstructionRaVirtualization<F: Field> {
    symbolic: relations::instruction::RaVirtualization,
    dimensions: InstructionRaVirtualizationDimensions,
    gamma: F,
    /// The stage-5 instruction address point, chunked into the per-chunk committed
    /// opening points.
    instruction_address: Vec<F>,
    /// The stage-5 instruction read-RAF cycle that `EqCycle` compares against.
    instruction_read_raf_cycle: Vec<F>,
    committed_chunk_bits: usize,
}

impl<F: Field> InstructionRaVirtualization<F> {
    pub fn new(
        dimensions: InstructionRaVirtualizationDimensions,
        gamma: F,
        instruction_address: Vec<F>,
        instruction_read_raf_cycle: Vec<F>,
        committed_chunk_bits: usize,
    ) -> Self {
        Self {
            symbolic: relations::instruction::RaVirtualization::new(dimensions),
            dimensions,
            gamma,
            instruction_address,
            instruction_read_raf_cycle,
            committed_chunk_bits,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionRaVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for InstructionRaVirtualization<F> {
    type Symbolic = relations::instruction::RaVirtualization;
    type Inputs<C> = InstructionRaVirtualizationInputClaims<C>;
    type Outputs<C> = InstructionRaVirtualizationOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &InstructionRaVirtualizationInputClaims<C>,
    ) -> Result<InstructionRaVirtualizationOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let committed_instruction_ra =
            committed_address_chunks(&self.instruction_address, self.committed_chunk_bits)
                .into_iter()
                .map(|chunk| [chunk.as_slice(), r_cycle.as_slice()].concat())
                .collect();
        Ok(InstructionRaVirtualizationOutputClaims {
            committed_instruction_ra,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::Gamma,
            ) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &InstructionRaVirtualizationInputClaims<C>,
        outputs: Option<&InstructionRaVirtualizationOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimDerived { id: *id })?;
        let JoltDerivedId::InstructionRaVirtualization(InstructionRaVirtualizationPublic::EqCycle) =
            id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let log_t = self.dimensions.log_t();
        let point = outputs
            .committed_instruction_ra
            .first()
            .map(GetPoint::point)
            .ok_or_else(|| {
                public_input_failed("instruction RA virtualization produced no openings")
            })?;
        let r_cycle = point.get(point.len() - log_t..).ok_or_else(|| {
            public_input_failed("instruction RA opening point shorter than log_t")
        })?;
        try_eq_mle(&self.instruction_read_raf_cycle, r_cycle).map_err(public_input_failed)
    }
}
