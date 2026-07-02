//! RAM `ra` virtualization symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    committed_ram_ra_product, ram_ra_claim_reduction, RamRaVirtualizationDimensions,
};
use crate::protocols::jolt::{JoltExpr, JoltRelationId, RamRaVirtualizationPublic};
use crate::SymbolicSumcheck;
use crate::{derived, opening, InputClaims, OutputClaims};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRaVirtualization)]
pub struct RamRaVirtualizationOutputClaims<C> {
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// The single reduced `RamRa` opening from the stage-5 RAM RA claim reduction.
#[derive(Clone, Debug, InputClaims)]
pub struct RamRaVirtualizationInputClaims<C> {
    #[opening(RamRa, from = RamRaClaimReduction)]
    pub ram_ra_reduced: C,
}

/// The RAM `ra` virtualization sumcheck: equates the reduced `ra` opening on the
/// input side with the product of the committed per-`d` `ra` openings, weighted
/// by the cycle-`eq` public, on the output side.
pub struct RaVirtualization {
    shape: RamRaVirtualizationDimensions,
}

impl SymbolicSumcheck for RaVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = RamRaVirtualizationDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RamRaVirtualizationInputClaims<C>;
    type Outputs<C> = RamRaVirtualizationOutputClaims<C>;

    fn new(shape: RamRaVirtualizationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRaVirtualization
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_ra_polys() + 1
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(ram_ra_claim_reduction())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(RamRaVirtualizationPublic::EqCycle) * committed_ram_ra_product(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ram::committed_ram_ra;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn ra_virtualization_dimensions(committed_ra_polys: usize) -> RamRaVirtualizationDimensions {
        RamRaVirtualizationDimensions::new(5, committed_ra_polys)
    }

    #[test]
    fn ra_virtualization_evaluates_like_core_formula() {
        let dimensions = ra_virtualization_dimensions(3);
        let relation = RaVirtualization::new(dimensions);

        let reduced = Fr::from_u64(3);
        let committed = [Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(11)];
        let eq_cycle = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == committed_ram_ra(0) => committed[0],
                id if id == committed_ram_ra(1) => committed[1],
                id if id == committed_ram_ra(2) => committed[2],
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::RamRaVirtualization(RamRaVirtualizationPublic::EqCycle) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(input, reduced);
        assert_eq!(
            output,
            eq_cycle * committed[0] * committed[1] * committed[2]
        );
    }

    #[test]
    fn ra_virtualization_supports_empty_ra_product() {
        let relation = RaVirtualization::new(ra_virtualization_dimensions(0));

        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![ram_ra_claim_reduction()]
        );
    }

    #[test]
    fn ra_virtualization_symbolic_matches_dependencies() {
        let relation = RaVirtualization::new(ra_virtualization_dimensions(3));

        assert_eq!(RaVirtualization::id(), JoltRelationId::RamRaVirtualization);
        assert_eq!(relation.rounds(), ra_virtualization_dimensions(3).log_t());
        assert_eq!(
            relation.degree(),
            ra_virtualization_dimensions(3).num_committed_ra_polys() + 1
        );
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                ram_ra_claim_reduction(),
                committed_ram_ra(0),
                committed_ram_ra(1),
                committed_ram_ra(2),
            ]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle)]
        );
    }
}
