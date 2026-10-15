//! Product-lane geometry shared by the composed Spartan relations and kernels.

#[cfg(feature = "field-inline")]
use crate::protocols::field_inline::geometry::product::selected_product_lanes;
use crate::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;

pub const SPARTAN_PRODUCT_BASE_LANES: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;

#[cfg(feature = "field-inline")]
pub const SPARTAN_PRODUCT_FIELD_INLINE_LANES: usize = selected_product_lanes().len();

#[cfg(not(feature = "field-inline"))]
pub const SPARTAN_PRODUCT_FIELD_INLINE_LANES: usize = 0;

pub const SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE: usize =
    SPARTAN_PRODUCT_BASE_LANES + SPARTAN_PRODUCT_FIELD_INLINE_LANES;
// The weighting kernel and both factors have degree at most domain_size - 1.
pub const SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE: usize =
    3 * (SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE - 1);
