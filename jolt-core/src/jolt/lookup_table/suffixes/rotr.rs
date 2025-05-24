use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum RotrSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for RotrSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let total_bits = b.len();
        let word_size = total_bits / 2;
        let (x, y) = b.uninterleave();

        // Convert to u32 for bit operations
        let x_val = u32::from(x);
        let y_val = u32::from(y);

        // Extract the current bit (LSB)
        let x_current = x_val & 1;
        let y_current = y_val & 1;

        // Get suffix (bits after current position, i.e., less significant bits)
        let y_suffix = y_val >> 1;
        let suffix_len = word_size.saturating_sub(1);

        if y_current == 1 {
            // y_j = 1: count ones in suffix
            let ones_count = y_suffix.count_ones();
            x_current * (1 << ones_count)
        } else {
            // y_j = 0: count zeros in suffix
            // Mask to only consider actual suffix bits
            let mask = if suffix_len >= 32 {
                u32::MAX
            } else {
                (1u32 << suffix_len) - 1
            };
            let zeros_count = (!y_suffix & mask).count_ones();
            x_current * (1 << (word_size - 1 - zeros_count as usize))
        }
    }
}
