use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// `L_s·P_s` for the store-data shift tables: the suffix-resident low bits of
/// the left operand, scaled by `2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))`.
pub enum ShiftDataSuffix<const XLEN: usize, const EIGHTHS: usize> {}

impl<const XLEN: usize, const EIGHTHS: usize> SparseDenseSuffix for ShiftDataSuffix<XLEN, EIGHTHS> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (xb, yb) = b.uninterleave();
        let lane_bits = EIGHTHS * (XLEN / 8);
        let lane = if lane_bits >= 64 {
            u64::from(xb)
        } else {
            u64::from(xb) & ((1u64 << lane_bits) - 1)
        };
        let offset = u64::from(yb) & (8 - EIGHTHS as u64);
        lane << ((XLEN / 8) as u64 * offset)
    }
}
