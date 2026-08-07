use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

pub enum LaneExtractValueSuffix {}
pub enum LaneExtractStraddleSuffix {}

fn suffix_state(b: LookupBits) -> (u128, u128, u128) {
    let (mut x, mut y) = b.uninterleave();
    let mut packed = 0u128;
    let mut top_count = 0u128;
    let mut signed_weight = 0u128;
    let mut previous_mask = 0u128;
    while !x.is_empty() {
        let x_i = u128::from(x.pop_msb());
        let y_i = u128::from(y.pop_msb());
        packed = packed * (1 + y_i) + x_i * y_i;
        let top = x_i * y_i * (1 - previous_mask);
        top_count += top;
        signed_weight = signed_weight * (1 + y_i) + top;
        previous_mask = y_i;
    }
    (packed, top_count, signed_weight)
}

impl SparseDenseSuffix for LaneExtractValueSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (packed, top_count, signed_weight) = suffix_state(b);
        (packed + (1u128 << XLEN) * top_count - 2 * signed_weight) as u64
    }
}

impl SparseDenseSuffix for LaneExtractStraddleSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (mut x, mut y) = b.uninterleave();
        if x.is_empty() {
            return 0;
        }
        let first_selected = u128::from(x.pop_msb()) * u128::from(y.pop_msb());
        let selected_below = u128::from(y).count_ones();
        (first_selected * ((1u128 << XLEN) - (2u128 << selected_below))) as u64
    }
}
