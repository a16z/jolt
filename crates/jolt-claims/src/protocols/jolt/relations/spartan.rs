//! Spartan symbolic sumcheck relations.

use jolt_field::RingCore;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::{
    branch_flag_product, is_first_in_sequence_shift, is_noop_shift, is_virtual_shift,
    jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_first_in_sequence_outer, next_is_noop_product, next_is_virtual_outer, next_pc_outer,
    next_unexpanded_pc_outer, outer_opening, outer_uniskip_opening, pc_shift,
    product_outer_opening, product_should_branch_outer_opening, product_should_jump_outer_opening,
    product_tau_kernel, product_uniskip_opening, product_uniskip_weight, product_weight,
    right_instruction_input_product, unexpanded_pc_shift, SpartanOuterDimensions,
    SpartanProductDimensions, SHIFT_DEGREE,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltDerivedId, JoltRelationId, JoltSumcheckSpec,
    SpartanOuterPublic, SpartanShiftChallenge, SpartanShiftPublic, TraceDimensions,
};
use crate::{challenge, opening, derived, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced product-remainder openings (the eight virtualized instruction-product
/// operands and flags), all sharing the single product opening point. Generic over
/// the cell (`F` on the wire / serialized proof form, `OpeningClaim<F>` on the
/// clear path). Field order is the canonical Fiat-Shamir order and must match
/// [`spartan::product_remainder_output_openings`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanProductVirtualization)]
pub struct ProductRemainderOutputClaims<C> {
    #[opening(LeftInstructionInput)]
    pub left_instruction_input: C,
    #[opening(RightInstructionInput)]
    pub right_instruction_input: C,
    #[opening(OpFlags(CircuitFlags::Jump))]
    pub jump_flag: C,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD))]
    pub write_lookup_output_to_rd: C,
    #[opening(LookupOutput)]
    pub lookup_output: C,
    #[opening(InstructionFlags(InstructionFlags::Branch))]
    pub branch_flag: C,
    #[opening(NextIsNoop)]
    pub next_is_noop: C,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub virtual_instruction: C,
}

/// Consumed product-remainder input: the product uni-skip's reduced opening. The
/// relation reads only this value (its output point comes from its own sumcheck
/// point), so the input point is left empty. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct ProductRemainderInputClaims<C> {
    #[opening(UnivariateSkip, from = SpartanProductVirtualization)]
    pub product_uniskip: C,
}

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

/// The Spartan outer univariate-skip sumcheck (first round). Symbolic-only: the
/// concrete uni-skip verification is special-cased in the verifier's stage 1.
pub struct OuterUniskip {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;

    fn new(shape: SpartanOuterDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanOuter
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.uniskip_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(outer_uniskip_opening())
    }
}

/// The Spartan outer remainder sumcheck: the quadratic R1CS form over the outer
/// R1CS-input openings, weighted by the `SpartanOuterPublic` coefficients.
pub struct OuterRemainder {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;

    fn new(shape: SpartanOuterDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanOuter
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.remainder_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(outer_uniskip_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let mut output = JoltExpr::zero();

        for (left_index, left_variable) in self.shape.variables().iter().copied().enumerate() {
            for (right_index, right_variable) in self.shape.variables().iter().copied().enumerate()
            {
                output = output
                    + derived(JoltDerivedId::from(
                        SpartanOuterPublic::QuadraticCoefficient {
                            left: left_index,
                            right: right_index,
                        },
                    )) * opening(outer_opening(left_variable))
                        * opening(outer_opening(right_variable));
            }
        }

        if self.shape.include_linear_terms() {
            for (index, variable) in self.shape.variables().iter().copied().enumerate() {
                output = output
                    + derived(JoltDerivedId::from(SpartanOuterPublic::LinearCoefficient(
                        index,
                    ))) * opening(outer_opening(variable));
            }
        }

        if self.shape.include_constant_term() {
            output = output + derived(JoltDerivedId::from(SpartanOuterPublic::ConstantCoefficient));
        }

        output
    }
}

/// The Spartan product univariate-skip sumcheck (first round). Symbolic-only:
/// special-cased in the verifier's stage 2.
pub struct ProductUniskip {
    shape: SpartanProductDimensions,
}

impl SymbolicSumcheck for ProductUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanProductDimensions;

    fn new(shape: SpartanProductDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanProductVirtualization
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.uniskip_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        product_uniskip_weight(0) * opening(product_outer_opening())
            + product_uniskip_weight(1) * opening(product_should_branch_outer_opening())
            + product_uniskip_weight(2) * opening(product_should_jump_outer_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(product_uniskip_opening())
    }
}

/// The Spartan product remainder sumcheck: the `tau_kernel * left * right`
/// virtualization form over the product-remainder openings.
pub struct ProductRemainder {
    shape: SpartanProductDimensions,
}

impl SymbolicSumcheck for ProductRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanProductDimensions;

    fn new(shape: SpartanProductDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanProductVirtualization
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.remainder_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(product_uniskip_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let left = product_weight(0) * opening(left_instruction_input_product())
            + product_weight(1) * opening(lookup_output_product())
            + product_weight(2) * opening(jump_flag_product());
        let right = product_weight(0) * opening(right_instruction_input_product())
            + product_weight(1) * opening(branch_flag_product())
            + product_weight(2) * (JoltExpr::one() - opening(next_is_noop_product()));

        product_tau_kernel() * left * right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{
        JoltChallengeId, JoltDerivedId, SpartanProductVirtualizationPublic,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

    #[test]
    fn product_remainder_evaluates_like_core_formula() {
        let relation = ProductRemainder::new(SpartanProductDimensions::new(7));

        let left_input = Fr::from_u64(2);
        let lookup_output = Fr::from_u64(3);
        let jump = Fr::from_u64(5);
        let right_input = Fr::from_u64(7);
        let branch = Fr::from_u64(11);
        let next_is_noop = Fr::from_u64(13);
        let weights = [Fr::from_u64(17), Fr::from_u64(19), Fr::from_u64(23)];
        let tau_kernel = Fr::from_u64(29);
        let zero = Fr::from_u64(0);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == left_instruction_input_product() => left_input,
                id if id == lookup_output_product() => lookup_output,
                id if id == jump_flag_product() => jump,
                id if id == right_instruction_input_product() => right_input,
                id if id == branch_flag_product() => branch,
                id if id == next_is_noop_product() => next_is_noop,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::SpartanProductVirtualization(
                    SpartanProductVirtualizationPublic::LagrangeWeight(index),
                ) => weights[index],
                JoltDerivedId::SpartanProductVirtualization(
                    SpartanProductVirtualizationPublic::TauKernel,
                ) => tau_kernel,
                _ => zero,
            },
        );

        assert_eq!(
            output,
            tau_kernel
                * (weights[0] * left_input + weights[1] * lookup_output + weights[2] * jump)
                * (weights[0] * right_input
                    + weights[1] * branch
                    + weights[2] * (Fr::from_u64(1) - next_is_noop))
        );
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
