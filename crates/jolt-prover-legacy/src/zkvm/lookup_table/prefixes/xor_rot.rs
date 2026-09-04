use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum XorRotPrefix<const XLEN: usize, const ROTATION: u32> {}

impl<const XLEN: usize, const ROTATION: u32, F: JoltField> SparseDensePrefix<F>
    for XorRotPrefix<XLEN, ROTATION>
{
    // Note: This function only works correctly for XLEN=64
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let suffix_len = LOG_K - j - b.len() - 1;
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

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
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
