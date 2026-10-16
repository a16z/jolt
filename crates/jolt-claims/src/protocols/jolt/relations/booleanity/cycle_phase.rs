use jolt_field::Ring;

use super::monolith::{BooleanityInputClaims, BooleanityOutputClaims};
use crate::opening;
use crate::protocols::jolt::geometry::booleanity::{
    booleanity_address_phase_opening, booleanity_cycle_output, BooleanityDimensions,
};
use crate::protocols::jolt::{BooleanityChallenge, JoltExpr, JoltRelationId};
use crate::{SumcheckChallenges, SymbolicSumcheck};

/// Fiat-Shamir challenge drawn by the cycle-phase split of the booleanity
/// sumcheck. As in the monolith, the `gamma` is built inside
/// `booleanity_cycle_output`, so this set is derived from `required_challenges()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BooleanityCyclePhaseChallenges<F> {
    #[challenge(BooleanityChallenge::Gamma)]
    pub gamma: F,
}

/// The cycle-phase split of the booleanity sumcheck: takes the
/// `BooleanityAddrClaim` opening as input and reduces to the boolean-constraint
/// output over the cycle variables.
#[derive(Clone)]
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

    fn rounds(&self) -> usize {
        self.shape.log_t
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(booleanity_address_phase_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        booleanity_cycle_output(self.shape)
    }
}
