use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// 1 if the first operand is (strictly) less than the
/// second operand; 0 otherwise.
pub enum LessThanSuffix {}

impl SparseDenseSuffix for LessThanSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (u64::from(x) < u64::from(y)).into()
    }
}
