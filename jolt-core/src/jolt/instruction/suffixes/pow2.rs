use crate::{subprotocols::sparse_dense_shout::LookupBits, utils::math::Math};

use super::SparseDenseSuffix;

pub enum Pow2Suffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for Pow2Suffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (_, shift) = b.split(WORD_SIZE.log_2());
        1 << u32::from(shift)
    }
}
