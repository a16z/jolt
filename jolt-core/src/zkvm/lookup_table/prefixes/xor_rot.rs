use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum XorRotPrefix<const XLEN: usize, const ROTATION: u32> {}

impl<const XLEN: usize, const ROTATION: u32, F: JoltField> SparseDensePrefix<F>
    for XorRotPrefix<XLEN, ROTATION>
{
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
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
        let suffix_len = current_suffix_len(j);

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
    ) -> PrefixCheckpoint<F> {
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
