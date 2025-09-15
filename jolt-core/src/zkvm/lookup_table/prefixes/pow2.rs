use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::utils::math::Math;
use crate::zkvm::instruction_lookups::read_raf_checking::current_suffix_len;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum Pow2Prefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for Pow2Prefix<XLEN> {
    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        if current_suffix_len(j) != 0 {
            // Handled by suffix
            return F::one();
        }

        // Shift amount is the last XLEN bits of b
        if b.len() >= XLEN.log_2() {
            return F::from_u64(1 << (b % XLEN));
        }

        let mut result = F::from_u64(1 << (b % XLEN));
        let mut num_bits = b.len();
        let mut shift = 1u64 << (1u64 << num_bits);
        result *= F::from_u64(1 + (shift - 1) * c as u64);

        // Shift amount is [c, b]
        if b.len() == XLEN.log_2() - 1 {
            return result;
        }

        // Shift amount is [(r, r_x), c, b]
        num_bits += 1;
        shift = 1 << (1 << num_bits);
        if let Some(r_x) = r_x {
            result *= F::one() + F::from_u64(shift - 1) * r_x;
        }

        result *= checkpoints[Prefixes::Pow2].unwrap_or(F::one());
        result
    }
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        if current_suffix_len(j) != 0 {
            // Handled by suffix
            return F::one();
        }

        // Shift amount is the last XLEN bits of b
        if b.len() >= XLEN.log_2() {
            return F::from_u64(1 << (b % XLEN));
        }

        let mut result = F::from_u64(1 << (b % XLEN));
        let mut num_bits = b.len();
        let mut shift = 1u64 << (1u64 << num_bits);
        result *= F::from_u64(1 + (shift - 1) * c as u64);

        // Shift amount is [c, b]
        if b.len() == XLEN.log_2() - 1 {
            return result;
        }

        // Shift amount is [(r, r_x), c, b]
        num_bits += 1;
        shift = 1 << (1 << num_bits);
        if let Some(r_x) = r_x {
            result *= F::one() + F::from_u64(shift - 1).mul_u128_mont_form(r_x);
        }

        result *= checkpoints[Prefixes::Pow2].unwrap_or(F::one());
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if current_suffix_len(j) != 0 {
            return Some(F::one()).into();
        }

        // r_y is the highest bit of the shift amount
        if j == 2 * XLEN - XLEN.log_2() {
            let shift = 1 << (XLEN / 2);
            return Some(F::one() + F::from_u64(shift - 1).mul_u128_mont_form(r_y)).into();
        }

        // r_x and r_y are bits in the shift amount
        if 2 * XLEN - j < XLEN.log_2() {
            let mut checkpoint = checkpoints[Prefixes::Pow2].unwrap();
            let shift = 1 << (1 << (2 * XLEN - j));
            checkpoint *= F::one() + F::from_u64(shift - 1).mul_u128_mont_form(r_x);
            let shift = 1 << (1 << (2 * XLEN - j - 1));
            checkpoint *= F::one() + F::from_u64(shift - 1).mul_u128_mont_form(r_y);
            return Some(checkpoint).into();
        }

        Some(F::one()).into()
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if current_suffix_len(j) != 0 {
            return Some(F::one()).into();
        }

        // r_y is the highest bit of the shift amount
        if j == 2 * XLEN - XLEN.log_2() {
            let shift = 1 << (XLEN / 2);
            return Some(F::one() + F::from_u64(shift - 1) * r_y).into();
        }

        // r_x and r_y are bits in the shift amount
        if 2 * XLEN - j < XLEN.log_2() {
            let mut checkpoint = checkpoints[Prefixes::Pow2].unwrap();
            let shift = 1 << (1 << (2 * XLEN - j));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_x;
            let shift = 1 << (1 << (2 * XLEN - j - 1));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_y;
            return Some(checkpoint).into();
        }

        Some(F::one()).into()
    }
}
