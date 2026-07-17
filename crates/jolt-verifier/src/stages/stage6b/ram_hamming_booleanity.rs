//! The stage 6 `RamHammingBooleanity` cycle-phase sumcheck instance.
//!
//! Proves the reduced RAM hamming-weight claim is boolean (`h² = h`) over the
//! trace. It consumes no opening claims (its input claim is the constant zero)
//! and produces the single `RamHammingWeight` opening; its only public,
//! `EqCycle`, ties the sumcheck point to the stage-1 Spartan-outer cycle binding.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamHammingBooleanityInputClaims, RamHammingBooleanityOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltDerivedId, JoltRelationId,
    RamHammingBooleanityPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

pub struct RamHammingBooleanity<F: Field> {
    symbolic: relations::ram::HammingBooleanity,
    trace_dimensions: TraceDimensions,
    /// The stage-1 Spartan-outer cycle binding that `EqCycle` compares the raw
    /// sumcheck point against.
    stage1_cycle_binding: Vec<F>,
}

impl<F: Field> RamHammingBooleanity<F> {
    pub fn new(trace_dimensions: TraceDimensions, stage1_cycle_binding: Vec<F>) -> Self {
        Self {
            symbolic: relations::ram::HammingBooleanity::new(trace_dimensions),
            trace_dimensions,
            stage1_cycle_binding,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamHammingBooleanity,
        reason: reason.to_string(),
    }
}

impl<F: Field> RamHammingBooleanity<F> {
    pub fn trace_dimensions(&self) -> TraceDimensions {
        self.trace_dimensions
    }

    pub fn stage1_cycle_binding(&self) -> &[F] {
        &self.stage1_cycle_binding
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamHammingBooleanity<F> {
    type Symbolic = relations::ram::HammingBooleanity;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RamHammingBooleanityInputClaims<Vec<F>>,
    ) -> Result<RamHammingBooleanityOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        Ok(RamHammingBooleanityOutputClaims {
            ram_hamming_weight: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &RamHammingBooleanityInputClaims<Vec<F>>,
        output_points: &RamHammingBooleanityOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamHammingBooleanity(RamHammingBooleanityPublic::EqCycle) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        // `cycle_opening_point` reverses the sumcheck point, so recover the raw
        // sumcheck point (what `EqCycle` compares against) by reversing back.
        let sumcheck_point = output_points
            .ram_hamming_weight()
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        try_eq_mle(&sumcheck_point, &self.stage1_cycle_binding).map_err(public_input_failed)
    }
}
