use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// `P_s = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` over the suffix bits of the
/// second operand. The suffix owns index bits [0, b.len()); with a boundary
/// inside the low offset bits this is the partial factor for the offset bits
/// it owns, and the OffsetScale/ShiftData prefixes supply the rest.
pub enum OffsetScaleSuffix<const EIGHTHS: usize> {}

impl SparseDenseSuffix for OffsetScaleSuffix<1> {
    fn suffix_mle(b: LookupBits) -> u64 {
        offset_scale_suffix(b, 1)
    }
}

impl SparseDenseSuffix for OffsetScaleSuffix<2> {
    fn suffix_mle(b: LookupBits) -> u64 {
        offset_scale_suffix(b, 2)
    }
}

impl SparseDenseSuffix for OffsetScaleSuffix<4> {
    fn suffix_mle(b: LookupBits) -> u64 {
        offset_scale_suffix(b, 4)
    }
}

fn offset_scale_suffix(b: LookupBits, eighths: u64) -> u64 {
    let bits: u128 = b.into();
    let mask = 8 - eighths;
    let mut shift = 0u64;
    // Offset bit y_i sits at even index position 2i.
    for i in 0..3 {
        if (mask >> i) & 1 == 1 && 2 * i < b.len() && (bits >> (2 * i)) & 1 == 1 {
            shift += ((XLEN / 8) << i) as u64;
        }
    }
    1 << shift
}
