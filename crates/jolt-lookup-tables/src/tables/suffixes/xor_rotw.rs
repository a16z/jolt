use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// XOR lower 32 bits of operands then rotate right by a constant.
pub enum XorRotWSuffix<const ROTATION: u32> {}

impl<const ROTATION: u32> SparseDenseSuffix for XorRotWSuffix<ROTATION> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let xor_result = (u64::from(x) as u32) ^ (u64::from(y) as u32);
        xor_result.rotate_right(ROTATION) as u64
    }
}
