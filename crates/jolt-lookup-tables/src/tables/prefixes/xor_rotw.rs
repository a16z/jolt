use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum XorRotWPrefix<const ROTATION: usize> {}

impl<const ROTATION: usize, F: Field> SparseDensePrefix<F> for XorRotWPrefix<ROTATION> {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start < XLEN {
            return F::zero();
        }

        let prefix_idx = match ROTATION {
            7 => Prefixes::XorRotW7,
            8 => Prefixes::XorRotW8,
            12 => Prefixes::XorRotW12,
            16 => Prefixes::XorRotW16,
            _ => unreachable!(),
        };

        let (x, y) = b.uninterleave();
        let xor_val = (u64::from(x) as u32) ^ (u64::from(y) as u32);

        let shift = if suffix_len / 2 >= ROTATION {
            suffix_len / 2 - ROTATION
        } else {
            32 + suffix_len / 2 - ROTATION
        };

        checkpoints[prefix_idx] + F::from_u32(xor_val.rotate_left(shift as u32))
    }
}
