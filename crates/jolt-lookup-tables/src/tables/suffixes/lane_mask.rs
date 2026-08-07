use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::tables::lane_mask::lane_value;

pub enum LaneMaskSuffix<const WIDTH_BYTES: usize> {}

impl<const WIDTH_BYTES: usize> SparseDenseSuffix for LaneMaskSuffix<WIDTH_BYTES> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.is_empty() {
            // Once every address bit is bound, no suffix factor remains.
            1
        } else {
            lane_value::<WIDTH_BYTES>(u64::from(b))
        }
    }
}
