use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// `σ·2^popcount(y)` where σ is the window sign (left operand's bit at the
/// right operand's most significant set bit).
pub enum WindowSignPow2Suffix {}

impl SparseDenseSuffix for WindowSignPow2Suffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        if y_val == 0 {
            0
        } else {
            let i = y_val.ilog2();
            ((x_val >> i) & 1) << y_val.count_ones()
        }
    }
}
