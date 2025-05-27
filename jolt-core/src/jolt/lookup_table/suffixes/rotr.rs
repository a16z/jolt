use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum RotrSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for RotrSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        debug_assert_eq!(x.len(), y.len());
        let n = x.len() as u32;
        let k = y.leading_ones();
        let (x, y) = (u32::from(x), u32::from(y));

        let first_part = (x & y) >> (n - k);
        let second_part = (x & !y) << (WORD_SIZE as u32 - n + k);
        first_part + second_part
    }
}
