use crate::utils::lookup_bits::LookupBits;
use crate::zkvm::lookup_table::lane_mask::lane_value;

use super::SparseDenseSuffix;

pub enum LaneMaskSuffix<const WIDTH_BYTES: usize> {}

impl<const WIDTH_BYTES: usize> SparseDenseSuffix for LaneMaskSuffix<WIDTH_BYTES> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            // Once every address bit is bound, no suffix factor remains.
            1
        } else {
            lane_value::<WIDTH_BYTES>((b & 7) as u64)
        }
    }
}
