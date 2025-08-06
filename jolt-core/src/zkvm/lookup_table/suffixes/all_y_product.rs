use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Computes the product of all y bits: ∏ᵢ yᵢ
/// Returns 1 only when all y bits are 1 (i.e., y = 2^n - 1)
pub enum AllYProductSuffix {}

impl SparseDenseSuffix for AllYProductSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_x, y) = b.uninterleave();
        let y_val = u64::from(y);
        let y_len = y.len();

        // Check if all y bits are 1 (y == 2^n - 1)
        if y_val == (1u64 << y_len) - 1 {
            1
        } else {
            0
        }
    }
}
