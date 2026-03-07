use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// XOR operands then rotate right by a constant.
pub enum XorRotSuffix<const ROTATION: u32> {}

impl<const ROTATION: u32> SparseDenseSuffix for XorRotSuffix<ROTATION> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let xor_result = u64::from(x) ^ u64::from(y);
        xor_result.rotate_right(ROTATION)
    }
}
