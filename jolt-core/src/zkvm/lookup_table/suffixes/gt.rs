use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// 1 if the first operand is (strictly) greater than the
/// second operand; 0 otherwise.
pub enum GreaterThanSuffix {}

impl SparseDenseSuffix for GreaterThanSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        (u32::from(x) > u32::from(y)).into()
    }
}
