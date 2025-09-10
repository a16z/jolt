use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Bitwise XOR with rotation by 32 suffix
pub enum XorRotSuffix {}

impl SparseDenseSuffix for XorRotSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        // println!("b Length is: {}", b.len());
        let (x, y) = b.uninterleave();
        let xor_result = u64::from(x) ^ u64::from(y);
        // Rotate right by 32 bits
        xor_result.rotate_right(32)
    }
}

// 100111101000011100001100111101010

// 101111100000011100001100111101010
