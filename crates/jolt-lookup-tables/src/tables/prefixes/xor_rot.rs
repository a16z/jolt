use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum XorRotPrefix<const ROTATION: u32> {}

impl<const ROTATION: u32, F: JoltField> SparseDensePrefix<F> for XorRotPrefix<ROTATION> {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        let xor_val = u64::from(x) ^ u64::from(y);

        // Each XOR bit at original position `p` maps to rotated position
        // `(p + ROTATION) % XLEN`. The phase bits correspond to original
        // positions starting at some offset. At binary points, we compute
        // the XOR and rotate the result into the correct output positions.
        //
        // The phase's x/y bits occupy positions that, after XOR and rotation,
        // need to be shifted to their final bit positions. The suffix bits
        // haven't been bound yet, so the phase XOR value gets rotated by
        // the appropriate amount.
        let rotation = ROTATION as usize;
        let shift = if suffix_len / 2 >= rotation {
            suffix_len / 2 - rotation
        } else {
            XLEN + suffix_len / 2 - rotation
        };

        checkpoints[Prefixes::xor_rot(ROTATION)] + F::from_u64(xor_val.rotate_left(shift as u32))
    }
}
