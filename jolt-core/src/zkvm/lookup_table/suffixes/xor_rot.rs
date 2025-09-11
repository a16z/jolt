use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Xor the operands of the suffix, and rotate right by a given constant value.
pub enum XorRotSuffix<const ROTATION: u32> {}

impl<const ROTATION: u32> SparseDenseSuffix for XorRotSuffix<ROTATION> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let xor_result = u64::from(x) ^ u64::from(y);
        xor_result.rotate_right(ROTATION)
    }
}
