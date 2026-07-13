//! Instruction RA-virtualization symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::instruction::{
    committed_instruction_ra_product, weighted_instruction_ra_sum,
    InstructionRaVirtualizationDimensions,
};
use crate::protocols::jolt::{
    InstructionRaVirtualizationChallenge, InstructionRaVirtualizationPublic, JoltExpr,
    JoltRelationId,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, InputClaims, OutputClaims, SumcheckChallenges};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(InstructionRaVirtualization)]
pub struct InstructionRaVirtualizationOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub committed_instruction_ra: Vec<C>,
}

/// The per-virtual reduced `InstructionRa` openings from the stage-5 instruction
/// read-RAF.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct InstructionRaVirtualizationInputClaims<C> {
    #[opening(InstructionRa, from = InstructionReadRaf)]
    pub instruction_ra: Vec<C>,
}

/// Fiat-Shamir challenge drawn by the instruction RA-virtualization sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct InstructionRaVirtualizationChallenges<F> {
    #[challenge(InstructionRaVirtualizationChallenge::Gamma)]
    pub gamma: F,
}

/// The instruction RA-virtualization sumcheck: relates the virtual
/// instruction-RA openings (folded by `gamma`) to the per-virtual products of
/// committed instruction-RA openings, weighted by the `EqCycle` public.
pub struct RaVirtualization {
    shape: InstructionRaVirtualizationDimensions,
}

impl SymbolicSumcheck for RaVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = InstructionRaVirtualizationDimensions;
    type Challenges<F> = InstructionRaVirtualizationChallenges<F>;
    type Inputs<C> = InstructionRaVirtualizationInputClaims<C>;
    type Outputs<C> = InstructionRaVirtualizationOutputClaims<C>;

    fn new(shape: InstructionRaVirtualizationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionRaVirtualization
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_per_virtual() + 1
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionRaVirtualizationChallenge::Gamma);
        weighted_instruction_ra_sum(self.shape, gamma)
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionRaVirtualizationChallenge::Gamma);
        let eq_cycle = derived(InstructionRaVirtualizationPublic::EqCycle);
        let mut output = JoltExpr::zero();
        for virtual_index in 0..self.shape.num_virtual_ra_polys() {
            output = output
                + eq_cycle.clone()
                    * gamma.clone().pow(virtual_index)
                    * committed_instruction_ra_product(self.shape, virtual_index);
        }
        output
    }
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{
        JoltChallengeId, JoltCommittedPolynomial, JoltDerivedId, JoltOpeningId, JoltPolynomialId,
        JoltVirtualPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn ra_virtualization_dimensions(
        num_virtual_ra_polys: usize,
        num_committed_per_virtual: usize,
    ) -> InstructionRaVirtualizationDimensions {
        InstructionRaVirtualizationDimensions::try_from((
            5,
            num_virtual_ra_polys,
            num_committed_per_virtual,
        ))
        .unwrap_or_else(|err| panic!("test RA virtualization dimensions should be valid: {err}"))
    }

    #[test]
    fn ra_virtualization_evaluates_like_core_formula() {
        let dimensions = ra_virtualization_dimensions(3, 2);
        let relation = RaVirtualization::new(dimensions);

        let virtual_ra = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        let committed_ra = [
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
            Fr::from_u64(23),
            Fr::from_u64(29),
        ];
        let gamma = Fr::from_u64(31);
        let eq_cycle = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Virtual(JoltVirtualPolynomial::InstructionRa(i)),
                    relation: JoltRelationId::InstructionReadRaf,
                } => virtual_ra[i],
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Committed(JoltCommittedPolynomial::InstructionRa(i)),
                    relation: JoltRelationId::InstructionRaVirtualization,
                } => committed_ra[i],
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::InstructionRaVirtualization(
                    InstructionRaVirtualizationPublic::EqCycle,
                ) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            virtual_ra[0] + gamma * virtual_ra[1] + gamma * gamma * virtual_ra[2]
        );
        assert_eq!(
            output,
            eq_cycle
                * (committed_ra[0] * committed_ra[1]
                    + gamma * committed_ra[2] * committed_ra[3]
                    + gamma * gamma * committed_ra[4] * committed_ra[5])
        );
    }

    #[test]
    fn ra_virtualization_symbolic_matches_dependencies() {
        let dimensions = ra_virtualization_dimensions(3, 2);
        let relation = RaVirtualization::new(dimensions);
        assert_eq!(
            RaVirtualization::id(),
            JoltRelationId::InstructionRaVirtualization
        );
        assert_eq!(relation.rounds(), dimensions.log_t());
        assert_eq!(
            relation.degree(),
            dimensions.num_committed_per_virtual() + 1
        );
    }
}
