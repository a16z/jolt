use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Helper suffix for left shift W variant.
/// Computes power of 2 based on leading ones count, truncated to 32 bits.
pub enum LeftShiftWHelperSuffix {}

impl SparseDenseSuffix for LeftShiftWHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        (1 << y.leading_ones()) as u32 as u64
    }
}
