use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// `L_s·P_s` for the store-data shift tables: the suffix-resident low bits of
/// the left operand, scaled by `2^(8·(y mod 8 & (8−EIGHTHS)))`.
pub enum ShiftDataSuffix<const EIGHTHS: usize> {}

impl SparseDenseSuffix for ShiftDataSuffix<1> {
    fn suffix_mle(b: LookupBits) -> u64 {
        shift_data_suffix(b, 1)
    }
}

impl SparseDenseSuffix for ShiftDataSuffix<2> {
    fn suffix_mle(b: LookupBits) -> u64 {
        shift_data_suffix(b, 2)
    }
}

impl SparseDenseSuffix for ShiftDataSuffix<4> {
    fn suffix_mle(b: LookupBits) -> u64 {
        shift_data_suffix(b, 4)
    }
}

fn shift_data_suffix(b: LookupBits, eighths: u64) -> u64 {
    let (xb, yb) = b.uninterleave();
    let lane_bits = eighths as usize * (XLEN / 8);
    let lane = if lane_bits >= 64 {
        u64::from(xb)
    } else {
        u64::from(xb) & ((1u64 << lane_bits) - 1)
    };
    let offset = u64::from(yb) & (8 - eighths);
    lane << ((XLEN / 8) as u64 * offset)
}
