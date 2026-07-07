//! The full (monolithic) booleanity symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::booleanity::{booleanity_cycle_output, BooleanityDimensions};
use crate::protocols::jolt::{BooleanityChallenge, JoltExpr, JoltRelationId};
use crate::{InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The committed per-family `Ra` openings produced by the cycle phase; every
/// opening shares the single booleanity opening point (`r_address ++ r_cycle`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(Booleanity)]
pub struct BooleanityOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// The `BooleanityAddrClaim` intermediate consumed from the address phase.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BooleanityInputClaims<C> {
    #[opening(BooleanityAddrClaim, from = Booleanity)]
    pub address_phase: C,
}

/// Fiat-Shamir challenge drawn by the full booleanity sumcheck. The `gamma`
/// folding the RA family is built inside the `booleanity_cycle_output` geometry
/// helper rather than appearing as a literal `challenge(..)` here, so this set is
/// derived from `required_challenges()`, not a textual scan of the expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct BooleanityChallenges<F> {
    #[challenge(BooleanityChallenge::Gamma)]
    pub gamma: F,
}

/// The full booleanity sumcheck over both the address and cycle variables:
/// asserts every one-hot `ra` opening is boolean (`ra^2 - ra == 0`), folded
/// across the RA family by `gamma` and weighted by the `EqAddressCycle` public.
/// Its input claim is zero — the boolean constraint sums to zero rather than
/// reducing a prior opening.
pub struct Booleanity {
    shape: BooleanityDimensions,
}

impl SymbolicSumcheck for Booleanity {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = BooleanityDimensions;
    type Challenges<F> = BooleanityChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = crate::NoOutputs<C>;

    fn new(shape: BooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn rounds(&self) -> usize {
        self.shape.sumcheck_rounds()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        booleanity_cycle_output(self.shape)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use crate::protocols::jolt::{
        BooleanityChallenge, BooleanityPublic, JoltChallengeId, JoltCommittedPolynomial,
        JoltDerivedId, JoltOpeningId,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions(instruction: usize, bytecode: usize, ram: usize) -> BooleanityDimensions {
        let layout = JoltRaPolynomialLayout::new(instruction, bytecode, ram).unwrap();
        BooleanityDimensions::new(layout, 5, 8)
    }

    #[test]
    fn booleanity_evaluates_like_core_formula() {
        let relation = Booleanity::new(dimensions(1, 1, 1));

        let instruction_ra = Fr::from_u64(3);
        let bytecode_ra = Fr::from_u64(5);
        let ram_ra = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let eq_address_cycle = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::InstructionRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    instruction_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::BytecodeRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    bytecode_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::RamRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    ram_ra
                }
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::Booleanity(BooleanityPublic::EqAddressCycle) => eq_address_cycle,
                _ => zero,
            },
        );

        let gamma_2 = gamma * gamma;
        let gamma_4 = gamma_2 * gamma_2;
        assert_eq!(
            output,
            eq_address_cycle
                * ((instruction_ra * instruction_ra - instruction_ra)
                    + gamma_2 * (bytecode_ra * bytecode_ra - bytecode_ra)
                    + gamma_4 * (ram_ra * ram_ra - ram_ra))
        );
    }

    #[test]
    fn booleanity_symbolic_matches_dependencies() {
        let relation = Booleanity::new(dimensions(1, 1, 1));
        assert_eq!(Booleanity::id(), JoltRelationId::Booleanity);
        assert_eq!(relation.rounds(), 13);
        assert_eq!(relation.degree(), 3);
    }

    /// The `gamma` is folded inside `booleanity_cycle_output`, so it never appears
    /// as a literal `challenge(..)` in this file. This guards that the
    /// `Challenges` struct's field set still matches the challenge the relation
    /// actually draws (per `required_challenges`).
    #[test]
    fn challenges_resolve_helper_built_gamma() {
        let challenges = BooleanityChallenges {
            gamma: Fr::from_u64(5),
        };

        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(BooleanityChallenge::Gamma)),
            Some(Fr::from_u64(5)),
        );
    }
}
