//! RAM RAF-evaluation symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    ram_address_spartan, ram_ra_raf_evaluation, RamRafEvaluationDimensions,
};
use crate::protocols::jolt::{JoltExpr, JoltRelationId, RamRafEvaluationPublic};
use crate::SymbolicSumcheck;
use crate::{constant, derived, opening, InputClaims, OutputClaims};

/// The produced RAM RAF `ram_ra` opening, sharing the single RAF opening point.
/// Generic over the cell (`F` on the wire / serialized proof form, `OpeningClaim<F>`
/// on the clear path).
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
#[derive(Clone, Debug, InputClaims)]
pub struct RamRafEvaluationInputClaims<C> {
    #[opening(RamAddress, from = SpartanOuter)]
    pub ram_address: C,
}

/// The RAM RAF-evaluation sumcheck: scales the Spartan RAM address opening by
/// `2^phase3_cycle_rounds` on the input side, and matches it against `ra`
/// weighted by the `UnmapAddress` public on the output side.
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

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        constant(F::pow2(self.shape.phase3_cycle_rounds())) * opening(ram_address_spartan())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(RamRafEvaluationPublic::UnmapAddress) * opening(ram_ra_raf_evaluation())
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltDerivedId, ReadWriteDimensions};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 4, 2, 1)
    }

    fn raf_evaluation_dimensions() -> RamRafEvaluationDimensions {
        RamRafEvaluationDimensions::try_from(read_write_dimensions())
            .expect("test RAM RAF evaluation dimensions should be valid")
    }

    #[test]
    fn raf_evaluation_evaluates_like_core_formula() {
        let dimensions = raf_evaluation_dimensions();
        let relation = RafEvaluation::new(dimensions);

        let address = Fr::from_u64(7);
        let ram_ra = Fr::from_u64(11);
        let unmap = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_address_spartan() => address,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra_raf_evaluation() => ram_ra,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::RamRafEvaluation(RamRafEvaluationPublic::UnmapAddress) => unmap,
                _ => zero,
            },
        );

        assert_eq!(input, address * Fr::from_u64(8));
        assert_eq!(output, unmap * ram_ra);
    }

    #[test]
    fn raf_evaluation_symbolic_matches_dependencies() {
        let relation = RafEvaluation::new(raf_evaluation_dimensions());

        assert_eq!(RafEvaluation::id(), JoltRelationId::RamRafEvaluation);
        assert_eq!(
            relation.rounds(),
            raf_evaluation_dimensions()
                .read_write()
                .raf_evaluation_rounds()
        );
        assert_eq!(relation.degree(), 2);
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![ram_address_spartan(), ram_ra_raf_evaluation()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress)]
        );
    }
}
