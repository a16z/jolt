use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum XorRotPrefix<const XLEN: usize, const ROTATION: u32> {}

impl<const XLEN: usize, const ROTATION: u32, F: Field> SparseDensePrefix<F>
    for XorRotPrefix<XLEN, ROTATION>
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
        let prefix_idx = match ROTATION {
            16 => Prefixes::XorRot16,
            24 => Prefixes::XorRot24,
            32 => Prefixes::XorRot32,
            63 => Prefixes::XorRot63,
            _ => unreachable!(),
        };
        let mut result = checkpoints[prefix_idx].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let xor_bit = (F::one() - r_x) * y + r_x * (F::one() - y);

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

        let (x, y) = b.uninterleave();

        let shift = if suffix_len as i32 / 2 - ROTATION as i32 >= 0 {
            suffix_len / 2 - ROTATION as usize
        } else {
            (XLEN as i32 + (suffix_len as i32 / 2 - ROTATION as i32)) as usize
        };

        result += F::from_u64((u64::from(x) ^ u64::from(y)).rotate_left(shift as u32));
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
        let prefix_idx = match ROTATION {
            16 => Prefixes::XorRot16,
            24 => Prefixes::XorRot24,
            32 => Prefixes::XorRot32,
            63 => Prefixes::XorRot63,
            _ => unreachable!(),
        };
        let original_pos = j / 2;
        let rotated_pos = (original_pos + ROTATION as usize) % XLEN;
        let shift = XLEN - 1 - rotated_pos;
        let updated = checkpoints[prefix_idx].unwrap_or(F::zero())
            + F::from_u64(1 << shift) * ((F::one() - r_x) * r_y + r_x * (F::one() - r_y));
        Some(updated).into()
    }
}
