use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum XorRotPrefix<const ROTATION: usize> {}

impl<const ROTATION: usize, F: Field> SparseDensePrefix<F> for XorRotPrefix<ROTATION> {
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

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        let prefix_idx = match ROTATION {
            16 => Prefixes::XorRot16,
            24 => Prefixes::XorRot24,
            32 => Prefixes::XorRot32,
            63 => Prefixes::XorRot63,
            _ => unimplemented!(),
        };
        let mut result = checkpoints[prefix_idx].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let xor_bit = (F::one() - r_x) * y + r_x * (F::one() - y);

            // Calculate where this bit ends up after rotation
            let original_pos = j / 2;
            let rotated_pos = (original_pos + ROTATION as usize) % XLEN;
            let shift = XLEN - 1 - rotated_pos;

            result += F::from_u64(1 << shift) * xor_bit;
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let xor_bit = (F::one() - x) * y_msb + x * (F::one() - y_msb);

            let original_pos = j / 2;
            let rotated_pos = (original_pos + ROTATION as usize) % XLEN;
            let shift = XLEN - 1 - rotated_pos;

            result += F::from_u64(1 << shift) * xor_bit;
        }

        // Remaining x and y bits
        let (x, y) = b.uninterleave();

        let shift = if suffix_len as i32 / 2 - ROTATION as i32 >= 0 {
            suffix_len / 2 - ROTATION as usize
        } else {
            (XLEN as i32 + (suffix_len as i32 / 2 - ROTATION as i32)) as usize
        };

        // Rotate left to position the XOR result bits correctly in the final output.
        result += F::from_u64((u64::from(x) ^ u64::from(y)).rotate_left(shift as u32));
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        let prefix_idx = match ROTATION {
            16 => Prefixes::XorRot16,
            24 => Prefixes::XorRot24,
            32 => Prefixes::XorRot32,
            63 => Prefixes::XorRot63,
            _ => unimplemented!(),
        };
        let original_pos = j / 2;
        let rotated_pos = (original_pos + ROTATION as usize) % XLEN;
        let shift = XLEN - 1 - rotated_pos;
        let updated = checkpoints[prefix_idx].unwrap_or(F::zero())
            + F::from_u64(1 << shift) * ((F::one() - r_x) * r_y + r_x * (F::one() - r_y));
        Some(updated).into()
    }
}
