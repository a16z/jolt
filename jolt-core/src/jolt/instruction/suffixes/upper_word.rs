use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

/// Shifts right by WORD_SIZE bits. Used by the MULHU instruction, which
/// multiplies two operands and returns the upper WORD_SIZE bits of the product.
pub enum UpperWordSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for UpperWordSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        (u64::from(b) >> WORD_SIZE) as u32
    }
}
