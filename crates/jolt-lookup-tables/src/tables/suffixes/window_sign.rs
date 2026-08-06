use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// The left operand's bit at the right operand's most significant set bit
/// (0 if the right operand is zero): the sign bit of a mask window.
pub enum WindowSignSuffix {}

impl SparseDenseSuffix for WindowSignSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        if y_val == 0 {
            0
        } else {
            let i = y_val.ilog2();
            (x_val >> i) & 1
        }
    }
}
