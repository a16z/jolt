use jolt_field::Ring;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, product_tau_kernel, product_uniskip_opening, product_weight,
    right_instruction_input_product, SpartanProductDimensions, PRODUCT_REMAINDER_DEGREE,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced product-remainder openings (the eight virtualized instruction-product
/// operands and flags), all sharing the single product opening point. Generic over
/// the opening cell (`F` for the serialized wire value, `Vec<F>` for the derived
/// opening point). Field declaration order is the canonical Fiat-Shamir order
/// (single-sourced via [`OutputClaims::canonical_order`]).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct ProductRemainderInputClaims<C> {
    #[opening(UnivariateSkip, from = SpartanProductVirtualization)]
    pub product_uniskip: C,
}

/// The Spartan product remainder sumcheck: the `tau_kernel * left * right`
/// virtualization form over the product-remainder openings.
#[derive(Clone)]
pub struct ProductRemainder {
    shape: SpartanProductDimensions,
}

impl ProductRemainder {
    /// The ordinary lanes' left-factor expression
    /// (`Σ_i product_weight(i) · left_i`). Exposed separately from
    /// [`output_expression`](SymbolicSumcheck::output_expression) — which is
    /// `tau_kernel · left · right` built from these factors — so the
    /// composed relation can extend each factor with the field-inline lanes'
    /// contributions without restating the ordinary lane table.
    pub fn left_factor_expression<F: Ring>(&self) -> JoltExpr<F> {
        product_weight(0) * opening(left_instruction_input_product())
            + product_weight(1) * opening(lookup_output_product())
            + product_weight(2) * opening(jump_flag_product())
    }

    /// The ordinary lanes' right-factor expression; see
    /// [`left_factor_expression`](Self::left_factor_expression).
    pub fn right_factor_expression<F: Ring>(&self) -> JoltExpr<F> {
        product_weight(0) * opening(right_instruction_input_product())
            + product_weight(1) * opening(branch_flag_product())
            + product_weight(2) * (JoltExpr::one() - opening(next_is_noop_product()))
    }
}

impl SymbolicSumcheck for ProductRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanProductDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = ProductRemainderInputClaims<C>;
    type Outputs<C> = ProductRemainderOutputClaims<C>;

    fn new(shape: SpartanProductDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanProductVirtualization
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        PRODUCT_REMAINDER_DEGREE
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(product_uniskip_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        product_tau_kernel() * self.left_factor_expression() * self.right_factor_expression()
    }
}
