use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Xor the operands of the suffix, while taking at most the lower 32-bit (or the size of them itself if they are less than 32), and rotate right by a given constant value.
pub enum XorRotWSuffix<const ROTATION: u32> {}

impl<const ROTATION: u32> SparseDenseSuffix for XorRotWSuffix<ROTATION> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        println!("suffix x is: {}", x);
        println!("suffix y is: {}", y);
        println!("suffix len is: {}", b.len() / 2);
        // Takes lower 32 bits only
        let x_32 = u64::from(x) as u32;
        let y_32 = u64::from(y) as u32;
        let xor_result = x_32 ^ y_32;
        // Rotate the 32-bit result and zero-extend to u64
        xor_result.rotate_right(ROTATION) as u64
    }
}
