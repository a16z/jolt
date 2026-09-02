use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum SignExtensionWSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignExtensionWSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            return 0;
        }

        let half = XLEN / 2;
        let (x, y) = b.uninterleave();
        let y_len = y.len();
        let mut fill = 0u128;
        if b.len() >= XLEN {
            let sign_bit = (u128::from(x) >> (half - 1)) & 1;
            if sign_bit == 0 {
                return 0;
            }
            fill = (1u128 << XLEN) - (1u128 << half);
        }

        let y_bits = u128::from(y);
        let count = y_len.min(half);
        let start_offset = y_len - count;
        let first_position = half - count;
        for offset in 0..count {
            let position = first_position + offset;
            if position == 0 {
                continue;
            }
            let y_bit = (y_bits >> (y_len - 1 - start_offset - offset)) & 1;
            fill += (1 - y_bit) << position;
        }
        fill as u64
    }
}
