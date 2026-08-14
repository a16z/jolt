//! The stage 6 `RamActivationBooleanity` cycle-phase sumcheck instance
//! (packed path).
//!
//! Binds the `Load`/`Store` activation columns over the trace and proves the
//! RAM activation sum Boolean (`(Load + Store)² = Load + Store`), producing
//! the two flag openings the stage-7 digit-zero baselines consume
//! (`specs/digit-zero-virtualization.md`). It consumes no opening claims (its
//! input claim is the constant zero); its only public, `EqCycle`, ties the
//! sumcheck point to the stage-1 Spartan-outer cycle binding.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamActivationBooleanityInputClaims, RamActivationBooleanityOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltDerivedId, JoltRelationId,
    RamActivationBooleanityPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[derive(Clone)]
pub struct RamActivationBooleanity<F: Field> {
    symbolic: relations::ram::ActivationBooleanity,
    trace_dimensions: TraceDimensions,
    /// The stage-1 Spartan-outer cycle binding that `EqCycle` compares the raw
    /// sumcheck point against.
    stage1_cycle_binding: Vec<F>,
}

impl<F: Field> RamActivationBooleanity<F> {
    pub fn new(trace_dimensions: TraceDimensions, stage1_cycle_binding: Vec<F>) -> Self {
        Self {
            symbolic: relations::ram::ActivationBooleanity::new(trace_dimensions),
            trace_dimensions,
            stage1_cycle_binding,
        }
    }

    pub fn trace_dimensions(&self) -> TraceDimensions {
        self.trace_dimensions
    }

    pub fn stage1_cycle_binding(&self) -> &[F] {
        &self.stage1_cycle_binding
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamActivationBooleanity,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamActivationBooleanity<F> {
    type Symbolic = relations::ram::ActivationBooleanity;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RamActivationBooleanityInputClaims<Vec<F>>,
    ) -> Result<RamActivationBooleanityOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        Ok(RamActivationBooleanityOutputClaims {
            load: opening_point.clone(),
            store: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &RamActivationBooleanityInputClaims<Vec<F>>,
        output_points: &RamActivationBooleanityOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamActivationBooleanity(RamActivationBooleanityPublic::EqCycle) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        // `cycle_opening_point` reverses the sumcheck point, so recover the raw
        // sumcheck point (what `EqCycle` compares against) by reversing back.
        let sumcheck_point = output_points
            .load()
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        try_eq_mle(&sumcheck_point, &self.stage1_cycle_binding).map_err(public_input_failed)
    }
}
