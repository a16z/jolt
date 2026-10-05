use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::booleanity::{booleanity_cycle_output, BooleanityDimensions};
use crate::protocols::jolt::{BooleanityChallenge, JoltExpr, JoltRelationId};
use crate::{InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The committed per-family `Ra` openings produced by the cycle phase; every
/// opening shares the single booleanity opening point (`r_address ++ r_cycle`).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        booleanity_cycle_output(self.shape)
    }
}
