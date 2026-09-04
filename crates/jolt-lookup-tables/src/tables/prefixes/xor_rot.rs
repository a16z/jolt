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
            2 => Prefixes::XorRot2,
            3 => Prefixes::XorRot3,
            8 => Prefixes::XorRot8,
            9 => Prefixes::XorRot9,
            19 => Prefixes::XorRot19,
            20 => Prefixes::XorRot20,
            21 => Prefixes::XorRot21,
            23 => Prefixes::XorRot23,
            25 => Prefixes::XorRot25,
            28 => Prefixes::XorRot28,
            36 => Prefixes::XorRot36,
            37 => Prefixes::XorRot37,
            39 => Prefixes::XorRot39,
            43 => Prefixes::XorRot43,
            44 => Prefixes::XorRot44,
            46 => Prefixes::XorRot46,
            49 => Prefixes::XorRot49,
            50 => Prefixes::XorRot50,
            54 => Prefixes::XorRot54,
            56 => Prefixes::XorRot56,
            58 => Prefixes::XorRot58,
            61 => Prefixes::XorRot61,
            62 => Prefixes::XorRot62,
            _ => unreachable!(),
        };

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
        let shift = if suffix_len / 2 >= ROTATION {
            suffix_len / 2 - ROTATION
        } else {
            XLEN + suffix_len / 2 - ROTATION
        };

        checkpoints[prefix_idx] + F::from_u64(xor_val.rotate_left(shift as u32))
    }
}
