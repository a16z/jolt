use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Returns the lower XLEN bits. Used to range-check values to be in
/// the range [0, 2^XLEN).
pub enum LowerWordSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for LowerWordSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        (u128::from(b) % (1 << XLEN)) as u64
    }
}
