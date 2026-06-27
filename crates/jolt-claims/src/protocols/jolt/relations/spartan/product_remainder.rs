//! Spartan product remainder symbolic sumcheck relation.

use jolt_field::RingCore;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, product_tau_kernel, product_uniskip_opening, product_weight,
    right_instruction_input_product, SpartanProductDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

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
    use crate::protocols::jolt::{JoltDerivedId, SpartanProductVirtualizationPublic};
    use jolt_field::{Fr, FromPrimitiveInt};

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
}
