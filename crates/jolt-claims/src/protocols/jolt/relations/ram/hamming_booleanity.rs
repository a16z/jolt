use core::marker::PhantomData;

use jolt_field::{JoltField, Ring};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::ram_hamming_weight;
use crate::protocols::jolt::{
    JoltExpr, JoltOpeningId, JoltRelationId, RamHammingBooleanityPublic, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{derived, opening, InputClaims, OutputClaims};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamHammingBooleanityInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for RamHammingBooleanityInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: JoltField> InputClaims<F> for RamHammingBooleanityInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        Vec::new()
    }

    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// The RAM Hamming-booleanity sumcheck: a degree-three output enforcing that the
/// Hamming-weight opening is boolean (`h^2 - h == 0`) at each cycle, weighted by
/// the cycle-`eq` public; no input claim.
#[derive(Clone)]
pub struct HammingBooleanity {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for HammingBooleanity {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RamHammingBooleanityInputClaims<C>;
    type Outputs<C> = RamHammingBooleanityOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamHammingBooleanity
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let eq_cycle = derived(RamHammingBooleanityPublic::EqCycle);
        let h = opening(ram_hamming_weight());
        eq_cycle * (h.clone() * h.clone() - h)
    }
}
