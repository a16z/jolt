use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::tables::lane_mask::lane_value;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LaneMaskPrefix<const WIDTH_BYTES: usize> {}

impl<const WIDTH_BYTES: usize, F: Field> SparseDensePrefix<F> for LaneMaskPrefix<WIDTH_BYTES> {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len != 0 {
            return F::one();
        }

        let prefix = match WIDTH_BYTES {
            0 => Prefixes::Pow2Lane,
            1 => Prefixes::LaneMaskB,
            2 => Prefixes::LaneMaskH,
            4 => Prefixes::LaneMaskW,
            _ => unreachable!("unsupported lane width"),
        };
        checkpoints[prefix] * F::from_u64(lane_value::<WIDTH_BYTES>(u64::from(b)))
    }
}
