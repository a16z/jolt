use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// `σ`: `x`'s bit at `y`'s most significant set bit, 0 if `y` is zero — the
/// sign bit of a mask window. Single source of the σ convention; every
/// window-sign prefix and suffix evaluates through this function.
#[inline]
pub(crate) fn window_sign_bit(x: u64, y: u64) -> u64 {
    if y == 0 {
        0
    } else {
        (x >> y.ilog2()) & 1
    }
}

/// The left operand's bit at the right operand's most significant set bit
/// (0 if the right operand is zero): the sign bit of a mask window.
pub enum WindowSignSuffix {}

impl SparseDenseSuffix for WindowSignSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        window_sign_bit(u64::from(x), u64::from(y))
    }
}
