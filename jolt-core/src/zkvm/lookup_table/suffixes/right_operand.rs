use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Computes the sum of y bits: Σ yᵢ * 2^(n-1-i)
pub enum RightOperandSuffix {}

impl SparseDenseSuffix for RightOperandSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_x, y) = b.uninterleave();
        u64::from(y)
    }
}
