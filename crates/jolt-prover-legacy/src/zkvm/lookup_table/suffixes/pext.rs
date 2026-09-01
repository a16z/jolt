use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// `pext(x, y)`: packs `x`'s bits at `y`'s set positions toward bit 0,
/// preserving order (the window's top bit lands at `popcount(y) − 1`). This
/// is the faithful form of the right-shift recurrence
/// `entry = entry·(1+y_i) + x_i·y_i`, valid for any mask shape.
#[inline]
pub(crate) fn pext(x: u64, y: u64) -> u64 {
    if y == 0 {
        return 0;
    }
    let tz = y.trailing_zeros();
    let normalized = y >> tz;
    if normalized & normalized.wrapping_add(1) == 0 {
        // Contiguous mask (every mask the window-mask tables produce):
        // extract is a shift plus truncate.
        return (x >> tz) & normalized;
    }
    // General mask: gather one bit per set position, lowest first.
    let mut bits = y;
    let mut out = 0u64;
    let mut k = 0;
    while bits != 0 {
        out |= ((x >> bits.trailing_zeros()) & 1) << k;
        k += 1;
        bits &= bits - 1;
    }
    out
}

/// Parallel bit extract: packs the left operand's bits at the right operand's
/// set positions, MSB-first (the faithful form of the right-shift recurrence
/// `entry = entry·(1+y_i) + x_i·y_i`, valid for any mask shape).
pub enum PextSuffix {}

impl SparseDenseSuffix for PextSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        pext(u64::from(x), u64::from(y))
    }
}
