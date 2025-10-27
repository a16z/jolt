use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// 1 if the first operand is (strictly) greater than the
/// second operand; 0 otherwise.
pub enum GreaterThanSuffix {}

impl SparseDenseSuffix for GreaterThanSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (u64::from(x) > u64::from(y)).into()
    }
}
