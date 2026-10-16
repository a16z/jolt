use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    ram_address_spartan, ram_ra_raf_evaluation, RamRafEvaluationDimensions,
};
use crate::protocols::jolt::{JoltExpr, JoltRelationId, RamRafEvaluationPublic};
use crate::SymbolicSumcheck;
use crate::{constant, derived, opening, InputClaims, OutputClaims};

/// The produced RAM RAF `ram_ra` opening, sharing the single RAF opening point.
/// Generic over the opening cell (`F` for the serialized wire value, `Vec<F>` for
/// the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRafEvaluation)]
pub struct RamRafEvaluationOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
}

/// The consumed RAM address opening from stage 1's outer sumcheck. The relation
/// reads only this value (its output point comes from its own sumcheck point), so
/// the input point is left empty. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RamRafEvaluationInputClaims<C> {
    #[opening(RamAddress, from = SpartanOuter)]
    pub ram_address: C,
}

/// The RAM RAF-evaluation sumcheck: scales the Spartan RAM address opening by
/// `2^phase3_cycle_rounds` on the input side, and matches it against `ra`
/// weighted by the `UnmapAddress` public on the output side.
#[derive(Clone)]
pub struct RafEvaluation {
    shape: RamRafEvaluationDimensions,
}

impl SymbolicSumcheck for RafEvaluation {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = RamRafEvaluationDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RamRafEvaluationInputClaims<C>;
    type Outputs<C> = RamRafEvaluationOutputClaims<C>;

    fn new(shape: RamRafEvaluationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRafEvaluation
    }

    fn rounds(&self) -> usize {
        self.shape.read_write().raf_evaluation_rounds()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        constant(F::pow2(self.shape.phase3_cycle_rounds())) * opening(ram_address_spartan())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        derived(RamRafEvaluationPublic::UnmapAddress) * opening(ram_ra_raf_evaluation())
    }
}
