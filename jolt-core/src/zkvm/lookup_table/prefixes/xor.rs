use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum XorPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for XorPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::Xor].unwrap_or(F::zero());

        // XOR high-order variables of x and y
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(1 << shift)
                * ((F::one() - F::from_u128_mont(r_x)) * y
                    + (F::one() - y).mul_u128_mont_form(r_x));
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(1 << shift) * ((F::one() - x) * y_msb + x * (F::one() - y_msb));
        }
        // XOR remaining x and y bits
        let (x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(2 * WORD_SIZE, j);
        result += F::from_u32((u32::from(x) ^ u32::from(y)) << (suffix_len / 2));

        result
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::Xor].unwrap_or(F::zero());

        // XOR high-order variables of x and y
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(1 << shift) * ((F::one() - r_x) * y + r_x * (F::one() - y));
        } else {
            let x = F::from_u32(c);
            let y_msb = F::from_u8(b.pop_msb());
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(1 << shift) * ((F::one() - x) * y_msb + x * (F::one() - y_msb));
        }
        // XOR remaining x and y bits
        let (x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(2 * WORD_SIZE, j);
        result += F::from_u32((u32::from(x) ^ u32::from(y)) << (suffix_len / 2));

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let shift = WORD_SIZE - 1 - j / 2;
        // checkpoint += 2^shift * ((1 - r_x) * r_y + r_x * (1 - r_y))
        let updated = checkpoints[Prefixes::Xor].unwrap_or(F::zero())
            + F::from_u32(1 << shift)
                * ((F::one() - F::from_u128_mont(r_x)).mul_u128_mont_form(r_y)
                    + (F::one() - F::from_u128_mont(r_y)).mul_u128_mont_form(r_x));
        Some(updated).into()
    }
}
