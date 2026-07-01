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
    InstructionRaVirtualizationChallenges, InstructionRaVirtualizationInputClaims,
    InstructionRaVirtualizationOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::committed_address_chunks, instruction::InstructionRaVirtualizationDimensions,
    },
    InstructionRaVirtualizationPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage5::Stage5ClearOutput;
use crate::VerifierError;

/// Wire the per-virtual reduced `InstructionRa` opening *values* from the stage-5
/// instruction read-RAF. (Verifier-side constructor for the moved
/// [`InstructionRaVirtualizationInputClaims`].)
pub fn instruction_ra_virtualization_input_values_from_upstream<F: Field>(
    stage5: &Stage5ClearOutput<F>,
) -> InstructionRaVirtualizationInputClaims<F> {
    InstructionRaVirtualizationInputClaims {
        instruction_ra: stage5
            .output_values
            .instruction_read_raf
            .instruction_ra
            .clone(),
    }
}

/// Wire the per-virtual reduced `InstructionRa` opening *points* from the stage-5
/// instruction read-RAF.
pub fn instruction_ra_virtualization_input_points_from_upstream<F: Field>(
    stage5: &Stage5ClearOutput<F>,
) -> InstructionRaVirtualizationInputClaims<Vec<F>> {
    InstructionRaVirtualizationInputClaims {
        instruction_ra: stage5
            .output_points
            .instruction_read_raf
            .instruction_ra()
            .to_vec(),
    }
}

pub struct InstructionRaVirtualization<F: Field> {
    symbolic: relations::instruction::RaVirtualization,
    dimensions: InstructionRaVirtualizationDimensions,
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
        instruction_address: Vec<F>,
        instruction_read_raf_cycle: Vec<F>,
        committed_chunk_bits: usize,
    ) -> Self {
        Self {
            symbolic: relations::instruction::RaVirtualization::new(dimensions),
            dimensions,
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &InstructionRaVirtualizationInputClaims<Vec<F>>,
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

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &InstructionRaVirtualizationInputClaims<Vec<F>>,
        output_points: &InstructionRaVirtualizationOutputClaims<Vec<F>>,
        _challenges: &InstructionRaVirtualizationChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::InstructionRaVirtualization(InstructionRaVirtualizationPublic::EqCycle) =
            id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let log_t = self.dimensions.log_t();
        let point = output_points
            .committed_instruction_ra()
            .first()
            .ok_or_else(|| {
                public_input_failed("instruction RA virtualization produced no openings")
            })?;
        let r_cycle = point.get(point.len() - log_t..).ok_or_else(|| {
            public_input_failed("instruction RA opening point shorter than log_t")
        })?;
        try_eq_mle(&self.instruction_read_raf_cycle, r_cycle).map_err(public_input_failed)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use core::num::NonZeroUsize;
    use jolt_field::Fr;
    use jolt_transcript::Transcript;

    fn relation(num_virtual_ra_polys: usize) -> InstructionRaVirtualization<Fr> {
        let dimensions = InstructionRaVirtualizationDimensions::new(
            3,
            NonZeroUsize::new(num_virtual_ra_polys).unwrap(),
            NonZeroUsize::new(1).unwrap(),
        )
        .unwrap();
        InstructionRaVirtualization::new(dimensions, Vec::new(), Vec::new(), 1)
    }

    // Inherits the default `draw_challenges`: the inline draw is
    // `challenge_scalar_powers(num_virtual_ra_polys())`, whose single squeeze's
    // degree-1 power equals that squeezed scalar — exactly what the default's one
    // `challenge_scalar` stores.
    #[test]
    fn default_draw_challenges_matches_inline_instruction_ra_gamma() {
        let relation = relation(2);
        let (inline_events, inline_gamma) = record(|t| t.challenge_scalar_powers(2)[1]);
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(draw_events, vec![DrawEvent::Squeeze(1)]);
        assert_eq!(challenges.gamma, inline_gamma);
    }

    // The only place the inline draw and the default could disagree is
    // `num_virtual_ra_polys() == 1`, where the inline `powers.get(1).unwrap_or(one)`
    // keeps `one` rather than the squeezed scalar. That disagreement is unobservable:
    // with a single virtual RA poly the gamma fold is `gamma^0`, so gamma is
    // structurally absent from both expressions and never resolved. Hence no override
    // is needed.
    #[test]
    fn single_virtual_ra_poly_omits_gamma_from_expressions() {
        let symbolic = relations::instruction::RaVirtualization::new(
            InstructionRaVirtualizationDimensions::new(
                3,
                NonZeroUsize::new(1).unwrap(),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap(),
        );
        assert!(symbolic.required_challenges::<Fr>().is_empty());
    }
}
