use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

#[derive(Default)]
pub struct UpperWordSuffix<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> SparseDenseSuffix for UpperWordSuffix<WORD_SIZE> {
    fn suffix_mle(&self, b: LookupBits) -> u32 {
        (u64::from(b) >> WORD_SIZE) as u32
    }
}
