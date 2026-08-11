use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// `P_s = 2^(8·(y mod 8 & (8−EIGHTHS)))` over the suffix bits of the second
/// operand (1 once the offset bits have been bound).
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
    if b.len() < 6 {
        return 1;
    }
    let (_, yb) = b.uninterleave();
    let offset = u64::from(yb) & (8 - eighths);
    1 << ((XLEN / 8) as u64 * offset)
}
