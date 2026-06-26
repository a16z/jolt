//! The stage 6 `RamHammingBooleanity` cycle-phase sumcheck instance.
//!
//! Proves the reduced RAM hamming-weight claim is boolean (`h² = h`) over the
//! trace. It consumes no opening claims (its input claim is the constant zero)
//! and produces the single `RamHammingWeight` opening; its only public,
//! `EqCycle`, ties the sumcheck point to the stage-1 Spartan-outer cycle binding.

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::relations;
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltOpeningId, JoltDerivedId, JoltRelationId,
    RamHammingBooleanityPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_claims_derive::OutputClaims;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{ConcreteSumcheck, GetPoint, InputClaims, OpeningClaim};
use crate::VerifierError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamHammingBooleanity)]
pub struct RamHammingBooleanityOutputClaims<C> {
    #[opening(RamHammingWeight)]
    pub ram_hamming_weight: C,
}

/// `RamHammingBooleanity` consumes no openings (its input claim is the constant
/// zero), so this carries only the cell marker. Hand-implements [`InputClaims`]
/// since the derive requires at least one `#[opening]` field.
pub struct RamHammingBooleanityInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for RamHammingBooleanityInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> RamHammingBooleanityInputClaims<OpeningClaim<F>> {
    pub fn from_upstream() -> Self {
        Self::default()
    }
}

impl<F: Field> InputClaims<F> for RamHammingBooleanityInputClaims<OpeningClaim<F>> {
    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

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

impl<F: Field> ConcreteSumcheck<F> for RamHammingBooleanity<F> {
    type Symbolic = relations::ram::HammingBooleanity;
    type Inputs<C> = RamHammingBooleanityInputClaims<C>;
    type Outputs<C> = RamHammingBooleanityOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RamHammingBooleanityInputClaims<C>,
    ) -> Result<RamHammingBooleanityOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        Ok(RamHammingBooleanityOutputClaims {
            ram_hamming_weight: opening_point,
        })
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &RamHammingBooleanityInputClaims<C>,
        outputs: Option<&RamHammingBooleanityOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimDerived { id: *id })?;
        let JoltDerivedId::RamHammingBooleanity(RamHammingBooleanityPublic::EqCycle) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        // `cycle_opening_point` reverses the sumcheck point, so recover the raw
        // sumcheck point (what `EqCycle` compares against) by reversing back.
        let sumcheck_point = outputs
            .ram_hamming_weight
            .point()
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        try_eq_mle(&sumcheck_point, &self.stage1_cycle_binding).map_err(public_input_failed)
    }
}
