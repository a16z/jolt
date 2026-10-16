use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum XorRotPrefix<const ROTATION: usize> {}

impl<const ROTATION: usize, F: JoltField> SparseDensePrefix<F> for XorRotPrefix<ROTATION> {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let prefix_idx = match ROTATION {
            16 => Prefixes::XorRot16,
            24 => Prefixes::XorRot24,
            32 => Prefixes::XorRot32,
            63 => Prefixes::XorRot63,
            _ => unreachable!(),
        };

        let (x, y) = b.uninterleave();
        let xor_val = u64::from(x) ^ u64::from(y);

        let shift = if suffix_len / 2 >= ROTATION {
            suffix_len / 2 - ROTATION
        } else {
            XLEN + suffix_len / 2 - ROTATION
        };

        checkpoints[prefix_idx] + F::from_u64(xor_val.rotate_left(shift as u32))
    }
}
