use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};
use crate::field::MontU128;
use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LowerHalfWordPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LowerHalfWordPrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let half_word_size = XLEN / 2;
        // Ignore high-order variables (those above the half-word boundary)
        if j < XLEN + half_word_size {
            return F::zero();
        }
        let mut result = checkpoints[Prefixes::LowerHalfWord].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = 2 * XLEN - j;
            let y_shift = 2 * XLEN - j - 1;
            result += F::from_u128(1u128 << x_shift).mul_u128_mont_form(r_x);
            result += F::from_u128(1u128 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = 2 * XLEN - j - 1;
            let y_shift = 2 * XLEN - j - 2;
            result += F::from_u128(1 << x_shift) * x;
            result += F::from_u128(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in low-order bits from `b`
        let suffix_len = current_suffix_len(j);
        result += F::from_u128(u128::from(b) << suffix_len);

        result
    }

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let half_word_size = XLEN / 2;
        // Ignore high-order variables (those above the half-word boundary)
        if j < XLEN + half_word_size {
            return F::zero();
        }
        let mut result = checkpoints[Prefixes::LowerHalfWord].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = 2 * XLEN - j;
            let y_shift = 2 * XLEN - j - 1;
            result += F::from_u128(1u128 << x_shift) * r_x;
            result += F::from_u128(1u128 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = 2 * XLEN - j - 1;
            let y_shift = 2 * XLEN - j - 2;
            result += F::from_u128(1 << x_shift) * x;
            result += F::from_u128(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in low-order bits from `b`
        let suffix_len = current_suffix_len(j);
        result += F::from_u128(u128::from(b) << suffix_len);

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let half_word_size = XLEN / 2;
        if j < XLEN + half_word_size {
            return None.into();
        }
        let x_shift = 2 * XLEN - j;
        let y_shift = 2 * XLEN - j - 1;
        let mut updated = checkpoints[Prefixes::LowerHalfWord].unwrap_or(F::zero());
        updated += F::from_u128(1 << x_shift).mul_u128_mont_form(r_x);
        updated += F::from_u128(1 << y_shift).mul_u128_mont_form(r_y);
        Some(updated).into()
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let half_word_size = XLEN / 2;
        if j < XLEN + half_word_size {
            return None.into();
        }
        let x_shift = 2 * XLEN - j;
        let y_shift = 2 * XLEN - j - 1;
        let mut updated = checkpoints[Prefixes::LowerHalfWord].unwrap_or(F::zero());
        updated += F::from_u128(1 << x_shift) * r_x;
        updated += F::from_u128(1 << y_shift) * r_y;
        Some(updated).into()
    }
}
