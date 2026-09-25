//! field_inline Spartan-outer produced claims.
//!
//! The field-inline extension appends five value/product columns to the composed
//! Spartan outer R1CS (`jolt-r1cs::constraints::jolt`); their openings are
//! produced by the same stage-1 remainder sumcheck as the ordinary RV64
//! openings and appended after them. There is no separate field-inline Spartan relation
//! object — the composed remainder is one sumcheck — so this module carries
//! only the typed claims struct for the appended segment.

use serde::{Deserialize, Serialize};

use crate::OutputClaims;

/// Produced field-inline Spartan-outer openings, in the appended-column order
/// (`geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS`): the five
/// value/product columns. All share the
/// stage-1 remainder opening point. Generic over the opening cell (`F` value /
/// `Vec<F>` point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
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
        };
        assert_eq!(outputs.canonical_order(), outer_output_openings());
    }
}
