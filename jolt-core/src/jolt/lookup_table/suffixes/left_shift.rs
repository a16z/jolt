use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

/// Left-shifts the left operand according to the bitmask given by
/// the right operand.
/// Performs shift to the left by the number of 1s in the right operand.
/// Removing bits of the left operand that have 1 in the right operand.
pub enum LeftShiftSuffix {}

impl SparseDenseSuffix for LeftShiftSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        let (x, y_u) = (u32::from(x), u32::from(y));
        // We remove bits of x that have 1 in y
        let x = x & !y_u;
        x.unbounded_shl(y.leading_ones())
    }
}
