use crate::utils::lookup_bits::LookupBits;

use super::window_sign::window_sign_bit;
use super::SparseDenseSuffix;

/// `σ·2^popcount(y)` where σ is the window sign (left operand's bit at the
/// right operand's most significant set bit).
pub enum WindowSignPow2Suffix {}

impl SparseDenseSuffix for WindowSignPow2Suffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        window_sign_bit(x_val, y_val) << y_val.count_ones()
    }
}
