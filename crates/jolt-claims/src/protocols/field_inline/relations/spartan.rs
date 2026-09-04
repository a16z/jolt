//! field_inline Spartan-outer produced claims.
//!
//! The field-inline extension appends 13 FR-local columns to the composed
//! Spartan outer R1CS (`jolt-r1cs::constraints::jolt`); their openings are
//! produced by the same stage-1 remainder sumcheck as the ordinary RV64
//! openings and appended after them. There is no separate FR Spartan relation
//! object — the composed remainder is one sumcheck — so this module carries
//! only the typed claims struct for the appended segment.

use serde::{Deserialize, Serialize};

use crate::protocols::field_inline::FieldInlineOpFlag;
use crate::OutputClaims;

/// Produced FR-local Spartan-outer openings, in the appended-column order
/// (`geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS`): the five
/// value/product columns, then the eight op-flag selectors. All share the
/// stage-1 remainder opening point. Generic over the opening cell (`F` value /
/// `Vec<F>` point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[protocol(field_inline)]
#[relation(FieldRegistersSpartanOuter)]
pub struct FieldRegistersSpartanOuterOutputClaims<C> {
    #[opening(FieldRs1Value)]
    pub rs1_value: C,
    #[opening(FieldRs2Value)]
    pub rs2_value: C,
    #[opening(FieldRdValue)]
    pub rd_value: C,
    #[opening(FieldProduct)]
    pub product: C,
    #[opening(FieldInvProduct)]
    pub inv_product: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::Add))]
    pub add: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::Sub))]
    pub sub: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::Mul))]
    pub mul: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::Inv))]
    pub inv: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::AssertEq))]
    pub assert_eq: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::LoadFromX))]
    pub load_from_x: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::StoreToX))]
    pub store_to_x: C,
    #[opening(FieldOpFlag(FieldInlineOpFlag::LoadImm))]
    pub load_imm: C,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::field_inline::geometry::spartan::outer_output_openings;
    use jolt_field::{Fr, Ring};

    /// The struct's field (declaration) order is the appended-column order the
    /// composed R1CS exposes, so the stage-1 absorb reproduces the column order
    /// byte-identically.
    #[test]
    fn claim_struct_field_order_matches_appended_column_order() {
        let value = Fr::from_u64(1);
        let outputs = FieldRegistersSpartanOuterOutputClaims::<Fr> {
            rs1_value: value,
            rs2_value: value,
            rd_value: value,
            product: value,
            inv_product: value,
            add: value,
            sub: value,
            mul: value,
            inv: value,
            assert_eq: value,
            load_from_x: value,
            store_to_x: value,
            load_imm: value,
        };
        assert_eq!(outputs.canonical_order(), outer_output_openings());
    }
}
