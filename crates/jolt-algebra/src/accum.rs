//! Signed accumulator reduction helpers.
//!
//! These live in a dedicated module so Akita-specific stage implementations can
//! share the same arithmetic helper without coupling to one another.

use jolt_field::Field;
use jolt_field::Unreduced;

#[inline]
/// Reduce separated positive and negative unreduced accumulators into one field
/// element.
pub fn reduce_signed_accum<E: Field + Unreduced>(pos: E::SmallProduct, neg: E::SmallProduct) -> E {
    E::reduce_small_product(pos) - E::reduce_small_product(neg)
}
