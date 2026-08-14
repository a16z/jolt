use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Parallel bit extract: packs the left operand's bits at the right operand's
/// set positions, MSB-first (the faithful form of the right-shift recurrence
/// `entry = entry·(1+y_i) + x_i·y_i`, valid for any mask shape).
pub enum PextSuffix {}

impl SparseDenseSuffix for PextSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let mut pext = 0u64;
        for i in (0..y.len()).rev() {
            if (y_val >> i) & 1 == 1 {
                pext = (pext << 1) | ((x_val >> i) & 1);
            }
        }
        pext
    }
}
