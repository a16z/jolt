use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum XorRotPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for XorRotPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::XorRot].unwrap_or(F::zero());

        // Compute XOR for the current high-order bit
        // This bit is at MLE position j/2 in the original XOR result
        // After rotating RIGHT by 32, it goes to MLE position (j/2 - 32 + XLEN) % XLEN
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let xor_bit = (F::one() - r_x) * y + r_x * (F::one() - y);

            // Calculate where this bit ends up after rotation
            let original_pos = j / 2;
            let rotated_pos = (original_pos + 32) % XLEN;
            let shift = XLEN - 1 - rotated_pos;

            result += F::from_u64(1 << shift) * xor_bit;
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let xor_bit = (F::one() - x) * y_msb + x * (F::one() - y_msb);

            let original_pos = j / 2;
            let rotated_pos = (original_pos + 32) % XLEN;
            let shift = XLEN - 1 - rotated_pos;

            result += F::from_u64(1 << shift) * xor_bit;
        }

        // XOR remaining x and y bits
        let (x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(j);

        // Calculate shift amount, handling negative values by wrapping around
        let shift_amount = if suffix_len as i32 / 2 - 32 >= 0 {
            (suffix_len / 2 - 32) as usize
        } else {
            // For negative shifts, wrap around: -1 -> XLEN-1, -2 -> XLEN-2, etc.
            (XLEN as i32 + (suffix_len as i32 / 2 - 32)) as usize
        };

        result += F::from_u64((u64::from(x) ^ u64::from(y)) << shift_amount);

        // XOR remaining x and y bits and compute their rotated contribution
        // let (x, y) = b.uninterleave();
        // println!("x and y in prefix are: {}, {}", x, y);
        // let suffix_len = current_suffix_len(j);
        // let xor_bits = u64::from(x) ^ u64::from(y);
        // result += F::from_u64(xor_bits.rotate_right(32 - 8));

        // These bits are at positions (j/2 + 1) through (XLEN - suffix_len/2 - 1) in the original XOR
        // After rotating right by 32, each bit needs to be placed individually at its new position
        // Rotate right by 32: position i -> position (i - 32 + XLEN) % XLEN
        // for i in 0..(XLEN - suffix_len / 2 - j / 2 - 1) {
        //     if (xor_bits >> i) & 1 == 1 {
        //         let original_mle_pos = j / 2 + 1 + i;
        //         // For rotate right by 32: new_pos = (old_pos + XLEN - 32) % XLEN
        //         // This handles wrap-around correctly (e.g., pos 0 -> 32, pos 31 -> 63)
        //         let rotated_mle_pos = (original_mle_pos + XLEN - 32) % XLEN;
        //         let shift = XLEN - 1 - rotated_mle_pos;
        //         result += F::from_u64(1 << shift);
        //     }
        // }

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        // The bit at MLE position j/2 after rotating RIGHT by 32 goes to position (j/2 - 32 + XLEN) % XLEN
        let original_pos = j / 2;
        let rotated_pos = (original_pos + XLEN - 32) % XLEN;
        let shift = XLEN - 1 - rotated_pos;
        // checkpoint += 2^shift * ((1 - r_x) * r_y + r_x * (1 - r_y))
        let updated = checkpoints[Prefixes::XorRot].unwrap_or(F::zero())
            + F::from_u64(1 << shift) * ((F::one() - r_x) * r_y + r_x * (F::one() - r_y));
        Some(updated).into()
    }
}
