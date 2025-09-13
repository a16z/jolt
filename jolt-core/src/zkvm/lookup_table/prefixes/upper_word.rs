use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

#[derive(Default)]
pub struct UpperWordPrefix<const XLEN: usize>;

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for UpperWordPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::UpperWord].unwrap_or(F::zero());
        // Ignore low-order variables
        if j >= WORD_SIZE {
            return result;
        }

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = WORD_SIZE - j;
            let y_shift = WORD_SIZE - j - 1;
            result += F::from_u64(1 << x_shift).mul_u128_mont_form(r_x);
            result += F::from_u64(1 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = WORD_SIZE - j - 1;
            let y_shift = WORD_SIZE - j - 2;
            result += F::from_u64(1 << x_shift) * x;
            result += F::from_u64(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in bits of `b` that fall in upper word
        let suffix_len = current_suffix_len(2 * WORD_SIZE, j);
        if suffix_len > WORD_SIZE {
            result += F::from_u64(u64::from(b) << (suffix_len - WORD_SIZE));
        } else {
            let (b_high, _) = b.split(WORD_SIZE - suffix_len);
            result += F::from_u64(u64::from(b_high));
        }

        result
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::UpperWord].unwrap_or(F::zero());
        // Ignore low-order variables
        if j >= XLEN {
            return result;
        }

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = XLEN - j;
            let y_shift = XLEN - j - 1;
            result += F::from_u64(1 << x_shift) * r_x;
            result += F::from_u64(1 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = XLEN - j - 1;
            let y_shift = XLEN - j - 2;
            result += F::from_u64(1 << x_shift) * x;
            result += F::from_u64(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in bits of `b` that fall in upper word
        let suffix_len = current_suffix_len(j);
        if suffix_len > XLEN {
            result += F::from_u64(u64::from(b) << (suffix_len - XLEN));
        } else {
            let (b_high, _) = b.split(XLEN - suffix_len);
            result += F::from_u64(u64::from(b_high));
        }

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j >= XLEN {
            return checkpoints[Prefixes::UpperWord].into();
        }
        let x_shift = XLEN - j;
        let y_shift = XLEN - j - 1;
        let updated = checkpoints[Prefixes::UpperWord].unwrap_or(F::zero())
            + F::from_u64(1 << x_shift).mul_u128_mont_form(r_x)
            + F::from_u64(1 << y_shift).mul_u128_mont_form(r_y);
        Some(updated).into()
    }
}
