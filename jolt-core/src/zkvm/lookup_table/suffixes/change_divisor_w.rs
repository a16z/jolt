use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum ChangeDivisorWSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for ChangeDivisorWSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let y_len = y.len().min(WORD_SIZE / 2);
        let (x, y) = (u64::from(x) as u32 as u64, u64::from(y) as u32 as u64);
        (((1u64 << y_len) - 1 == y) && x == 0).into()
    }
}
