use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum XorRotWPrefix<const XLEN: usize, const ROTATION: u32> {}

impl<const XLEN: usize, const ROTATION: u32, F: Field> SparseDensePrefix<F>
    for XorRotWPrefix<XLEN, ROTATION>
{
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let suffix_len = 2 * XLEN - j - b.len() - 1;
        if j < XLEN {
            return F::zero();
        }

        let prefix_idx = match ROTATION {
            7 => Prefixes::XorRotW7,
            8 => Prefixes::XorRotW8,
            12 => Prefixes::XorRotW12,
            16 => Prefixes::XorRotW16,
            _ => unreachable!(),
        };
        let mut result = checkpoints[prefix_idx].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let xor_bit = (F::one() - r_x) * y + r_x * (F::one() - y);
            let position = (j - XLEN) / 2;
            let mut rotated_position = (position + ROTATION as usize) % 32;
            rotated_position = 32 - 1 - rotated_position;
            result += F::from_u64(1 << rotated_position) * xor_bit;
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let xor_bit = (F::one() - x) * y_msb + x * (F::one() - y_msb);
            let position = (j - XLEN) / 2;
            let mut rotated_position = (position + ROTATION as usize) % 32;
            rotated_position = 32 - 1 - rotated_position;
            result += F::from_u64(1 << rotated_position) * xor_bit;
        }

        let (x, y) = b.uninterleave();
        let x_32 = u64::from(x) as u32;
        let y_32 = u64::from(y) as u32;
        let xor_result = x_32 ^ y_32;

        let shift = if suffix_len as i32 / 2 - ROTATION as i32 >= 0 {
            suffix_len / 2 - ROTATION as usize
        } else {
            (32_i32 + (suffix_len as i32 / 2 - ROTATION as i32)) as usize
        };

        let shifted = xor_result.rotate_left(shift as u32);
        result += F::from_u32(shifted);
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j >= XLEN {
            let prefix_idx = match ROTATION {
                7 => Prefixes::XorRotW7,
                8 => Prefixes::XorRotW8,
                12 => Prefixes::XorRotW12,
                16 => Prefixes::XorRotW16,
                _ => unreachable!(),
            };
            let original_pos = (j - XLEN) / 2;
            let rotated_pos = (original_pos + ROTATION as usize) % 32;
            let shift = 32 - 1 - rotated_pos;
            let updated = checkpoints[prefix_idx].unwrap_or(F::zero())
                + F::from_u64(1 << shift) * ((F::one() - r_x) * r_y + r_x * (F::one() - r_y));
            Some(updated).into()
        } else {
            Some(F::zero()).into()
        }
    }
}
