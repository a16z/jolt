use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Shifts right by XLEN bits. Used by the MULHU instruction, which
/// multiplies two operands and returns the upper XLEN bits of the product.
pub enum UpperWordSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for UpperWordSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        (u128::from(b) >> XLEN) as u64
    }
}
