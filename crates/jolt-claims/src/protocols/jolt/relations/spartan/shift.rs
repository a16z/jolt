//! Spartan shift symbolic sumcheck relation.

use jolt_field::RingCore;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::{
    is_first_in_sequence_shift, is_noop_shift, is_virtual_shift, next_is_first_in_sequence_outer,
    next_is_noop_product, next_is_virtual_outer, next_pc_outer, next_unexpanded_pc_outer, pc_shift,
    unexpanded_pc_shift, SHIFT_DEGREE,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, JoltSumcheckSpec, SpartanShiftChallenge, SpartanShiftPublic,
    TraceDimensions,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

/// Produced Spartan shift openings (the shifted unexpanded-PC / PC / virtual /
/// first-in-sequence / noop columns), all sharing the single shift opening point.
/// Generic over the cell.
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
#[derive(Clone, Debug, InputClaims)]
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

/// Fiat-Shamir challenge drawn by the Spartan shift sumcheck.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct SpartanShiftChallenges<F> {
    #[challenge(SpartanShiftChallenge::Gamma)]
    pub gamma: F,
}

/// The Spartan shift sumcheck: relates each `Next*` column from the outer
/// sumcheck (and `next_is_noop` from the product remainder) to the shifted
/// column at the same cycle, folded by `gamma` and weighted by the `EqPlusOne`
/// publics.
pub struct Shift {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for Shift {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = SpartanShiftChallenges<F>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanShift
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(SHIFT_DEGREE)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(SpartanShiftChallenge::Gamma);
        opening(next_unexpanded_pc_outer())
            + gamma.clone() * opening(next_pc_outer())
            + gamma.clone().pow(2) * opening(next_is_virtual_outer())
            + gamma.clone().pow(3) * opening(next_is_first_in_sequence_outer())
            + gamma.pow(4) * (JoltExpr::one() - opening(next_is_noop_product()))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(SpartanShiftChallenge::Gamma);
        derived(SpartanShiftPublic::EqPlusOneOuter)
            * (opening(unexpanded_pc_shift())
                + gamma.clone() * opening(pc_shift())
                + gamma.clone().pow(2) * opening(is_virtual_shift())
                + gamma.clone().pow(3) * opening(is_first_in_sequence_shift()))
            + derived(SpartanShiftPublic::EqPlusOneProduct)
                * gamma.pow(4)
                * (JoltExpr::one() - opening(is_noop_shift()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

    #[test]
    fn shift_evaluates_like_core_formula() {
        let relation = Shift::new(TraceDimensions::new(5));

        let next_unexpanded_pc = Fr::from_u64(3);
        let next_pc = Fr::from_u64(5);
        let next_virtual = Fr::from_u64(7);
        let next_first = Fr::from_u64(11);
        let next_noop = Fr::from_u64(13);
        let unexpanded_pc = Fr::from_u64(17);
        let pc = Fr::from_u64(19);
        let is_virtual = Fr::from_u64(23);
        let is_first = Fr::from_u64(29);
        let is_noop = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let eq_outer = Fr::from_u64(41);
        let eq_product = Fr::from_u64(43);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == next_unexpanded_pc_outer() => next_unexpanded_pc,
                id if id == next_pc_outer() => next_pc,
                id if id == next_is_virtual_outer() => next_virtual,
                id if id == next_is_first_in_sequence_outer() => next_first,
                id if id == next_is_noop_product() => next_noop,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == unexpanded_pc_shift() => unexpanded_pc,
                id if id == pc_shift() => pc,
                id if id == is_virtual_shift() => is_virtual,
                id if id == is_first_in_sequence_shift() => is_first,
                id if id == is_noop_shift() => is_noop,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::SpartanShift(SpartanShiftPublic::EqPlusOneOuter) => eq_outer,
                JoltDerivedId::SpartanShift(SpartanShiftPublic::EqPlusOneProduct) => eq_product,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            next_unexpanded_pc
                + gamma * next_pc
                + gamma_power(gamma, 2) * next_virtual
                + gamma_power(gamma, 3) * next_first
                + gamma_power(gamma, 4) * (Fr::from_u64(1) - next_noop)
        );
        assert_eq!(
            output,
            eq_outer
                * (unexpanded_pc
                    + gamma * pc
                    + gamma_power(gamma, 2) * is_virtual
                    + gamma_power(gamma, 3) * is_first)
                + eq_product * gamma_power(gamma, 4) * (Fr::from_u64(1) - is_noop)
        );
    }

    #[test]
    fn shift_symbolic_matches_dependencies() {
        let relation = Shift::new(TraceDimensions::new(5));
        assert_eq!(Shift::id(), JoltRelationId::SpartanShift);
        assert_eq!(
            relation.spec(),
            TraceDimensions::new(5).sumcheck(SHIFT_DEGREE)
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(SpartanShiftChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(SpartanShiftPublic::EqPlusOneOuter),
                JoltDerivedId::from(SpartanShiftPublic::EqPlusOneProduct),
            ]
        );
    }
}
