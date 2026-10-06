use jolt_field::Ring;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::SHIFT_DEGREE;
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, JoltVirtualPolynomial, SpartanShiftChallenge, SpartanShiftPublic,
    TraceDimensions, UnbatchedClaim, UnbatchedClaimExpr, UnbatchedRelation,
};
use crate::{InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// Produced Spartan shift openings (the shifted unexpanded-PC / PC / virtual /
/// first-in-sequence / noop columns), all sharing the single shift opening point.
/// Generic over the cell.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanShift)]
pub struct SpartanShiftOutputClaims<C> {
    #[opening(UnexpandedPC)]
    pub unexpanded_pc: C,
    #[opening(PC)]
    pub pc: C,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub is_virtual: C,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence))]
    pub is_first_in_sequence: C,
    #[opening(InstructionFlags(InstructionFlags::IsNoop))]
    pub is_noop: C,
}

/// Consumed shift openings: the `Next*` PC/flag columns from stage 1's outer
/// sumcheck and `next_is_noop` from stage 2's product remainder. Shift reads only
/// these values, so the input points are left empty. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct SpartanShiftInputClaims<C> {
    #[opening(NextUnexpandedPC, from = SpartanOuter)]
    pub next_unexpanded_pc: C,
    #[opening(NextPC, from = SpartanOuter)]
    pub next_pc: C,
    #[opening(NextIsVirtual, from = SpartanOuter)]
    pub next_is_virtual: C,
    #[opening(NextIsFirstInSequence, from = SpartanOuter)]
    pub next_is_first_in_sequence: C,
    #[opening(NextIsNoop, from = SpartanProductVirtualization)]
    pub next_is_noop: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct SpartanShiftChallenges<F> {
    #[challenge(SpartanShiftChallenge::Gamma)]
    pub gamma: F,
}

/// The Spartan shift sumcheck: relates each `Next*` column from the outer
/// sumcheck (and `next_is_noop` from the product remainder) to the shifted
/// column at the same cycle, folded by `gamma` and weighted by the `EqPlusOne`
/// publics.
#[derive(Clone)]
pub struct Shift {
    shape: TraceDimensions,
}

impl Shift {
    pub fn unbatched_relation() -> UnbatchedRelation {
        let v = UnbatchedClaimExpr::polynomial;
        let one = || UnbatchedClaimExpr::constant(1);
        UnbatchedRelation {
            output_relation: JoltRelationId::SpartanShift,
            gamma: SpartanShiftChallenge::Gamma.into(),
            claims: vec![
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: v(JoltVirtualPolynomial::NextUnexpandedPC),
                    output: v(JoltVirtualPolynomial::UnexpandedPC),
                    output_weight: SpartanShiftPublic::EqPlusOneOuter.into(),
                    offset: true,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: v(JoltVirtualPolynomial::NextPC),
                    output: v(JoltVirtualPolynomial::PC),
                    output_weight: SpartanShiftPublic::EqPlusOneOuter.into(),
                    offset: true,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: v(JoltVirtualPolynomial::NextIsVirtual),
                    output: v(JoltVirtualPolynomial::OpFlags(
                        CircuitFlags::VirtualInstruction,
                    )),
                    output_weight: SpartanShiftPublic::EqPlusOneOuter.into(),
                    offset: true,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: v(JoltVirtualPolynomial::NextIsFirstInSequence),
                    output: v(JoltVirtualPolynomial::OpFlags(
                        CircuitFlags::IsFirstInSequence,
                    )),
                    output_weight: SpartanShiftPublic::EqPlusOneOuter.into(),
                    offset: true,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanProductVirtualization,
                    input: one() - v(JoltVirtualPolynomial::NextIsNoop),
                    output: one()
                        - v(JoltVirtualPolynomial::InstructionFlags(
                            InstructionFlags::IsNoop,
                        )),
                    output_weight: SpartanShiftPublic::EqPlusOneProduct.into(),
                    offset: true,
                },
            ],
        }
    }
}

impl SymbolicSumcheck for Shift {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = SpartanShiftChallenges<F>;
    type Inputs<C> = SpartanShiftInputClaims<C>;
    type Outputs<C> = SpartanShiftOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanShift
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        SHIFT_DEGREE
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        Self::unbatched_relation().folded_input()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        Self::unbatched_relation().folded_output()
    }
}
