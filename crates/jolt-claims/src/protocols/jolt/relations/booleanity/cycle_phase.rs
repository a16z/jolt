//! The cycle-phase split of the booleanity symbolic sumcheck relation.

use jolt_field::RingCore;

use super::monolith::{BooleanityInputClaims, BooleanityOutputClaims};
use crate::opening;
use crate::protocols::jolt::geometry::booleanity::{
    booleanity_address_phase_opening, booleanity_cycle_output, BooleanityDimensions,
};
use crate::protocols::jolt::{BooleanityChallenge, JoltExpr, JoltRelationId, JoltSumcheckSpec};
use crate::{SumcheckChallenges, SymbolicSumcheck};

/// Fiat-Shamir challenge drawn by the cycle-phase split of the booleanity
/// sumcheck. As in the monolith, the `gamma` is built inside
/// `booleanity_cycle_output`, so this set is derived from `required_challenges()`.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct BooleanityCyclePhaseChallenges<F> {
    #[challenge(BooleanityChallenge::Gamma)]
    pub gamma: F,
}

/// The cycle-phase split of the booleanity sumcheck: takes the
/// `BooleanityAddrClaim` opening as input and reduces to the boolean-constraint
/// output over the cycle variables.
pub struct BooleanityCyclePhase {
    shape: BooleanityDimensions,
}

impl SymbolicSumcheck for BooleanityCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = BooleanityDimensions;
    type Challenges<F> = BooleanityCyclePhaseChallenges<F>;
    type Inputs<C> = BooleanityInputClaims<C>;
    type Outputs<C> = BooleanityOutputClaims<C>;

    fn new(shape: BooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.cycle_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(booleanity_address_phase_opening())
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
        BooleanityChallenge, BooleanityPublic, JoltChallengeId, JoltDerivedId, JoltSumcheckSpec,
    };
    use jolt_field::Fr;

    fn dimensions(instruction: usize, bytecode: usize, ram: usize) -> BooleanityDimensions {
        let layout = JoltRaPolynomialLayout::new(instruction, bytecode, ram).unwrap();
        BooleanityDimensions::new(layout, 5, 8)
    }

    #[test]
    fn booleanity_cycle_phase_symbolic_matches_dependencies() {
        let relation = BooleanityCyclePhase::new(dimensions(1, 1, 1));
        assert_eq!(BooleanityCyclePhase::id(), JoltRelationId::Booleanity);
        assert_eq!(relation.spec(), JoltSumcheckSpec::boolean(5, 3));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![booleanity_address_phase_opening()]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(BooleanityChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(BooleanityPublic::EqAddressCycle)]
        );
    }
}
