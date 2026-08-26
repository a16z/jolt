use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// `P_s = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` over the suffix bits of the
/// second operand. The suffix owns index bits [0, b.len()); with a boundary
/// inside the low offset bits this is the partial factor for the offset bits
/// it owns, and the OffsetScale/ShiftData prefixes supply the rest.
pub enum OffsetScaleSuffix<const XLEN: usize, const EIGHTHS: usize> {}

impl<const XLEN: usize, const EIGHTHS: usize> SparseDenseSuffix
    for OffsetScaleSuffix<XLEN, EIGHTHS>
{
    fn suffix_mle(b: LookupBits) -> u64 {
        let bits: u128 = b.into();
        let mask = (8 - EIGHTHS) as u64;
        let mut shift = 0u64;
        // Offset bit y_i sits at even index position 2i.
        for i in 0..3 {
            if (mask >> i) & 1 == 1 && 2 * i < b.len() && (bits >> (2 * i)) & 1 == 1 {
                shift += ((XLEN / 8) << i) as u64;
            }
        }
        1 << shift
    }
}
