use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Left-shifts x by the number of leading 1s in y, masking out matched bits.
pub enum LeftShiftSuffix {}

impl SparseDenseSuffix for LeftShiftSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let (x, y_u) = (u64::from(x), u64::from(y));
        let x = x & !y_u;
        x.unbounded_shl(y.leading_ones())
    }
}
