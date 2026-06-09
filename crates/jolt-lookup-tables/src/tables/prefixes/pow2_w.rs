use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum Pow2WPrefix {}

impl<F: Field> SparseDensePrefix<F> for Pow2WPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len != 0 {
            return F::one();
        }

        // pow2w computes 2^(shift mod 32). The shift is the last 5 interleaved bits.
        checkpoints[Prefixes::Pow2W] * F::from_u64(1u64 << (b & 0b11111))
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        if suffix_len != 0 {
            // Handled by suffix
            return F::one();
        }

        // Shift amount is the last 5 bits of b (for modulo 32)
        if b.len() >= 5 {
            return F::from_u64(1 << (b & (0b11111)));
        }

        let mut result = F::from_u64(1 << (b & (0b11111)));
        let mut num_bits = b.len();
        let mut shift = 1u64 << (1u64 << num_bits);
        result *= F::from_u64(1 + (shift - 1) * c as u64);

        // Shift amount is [c, b]
        if b.len() == 4 {
            // 5 - 1
            return result;
        }

        // Shift amount is [(r, r_x), c, b]
        num_bits += 1;
        shift = 1 << (1 << num_bits);
        if let Some(r_x) = r_x {
            result *= F::one() + F::from_u64(shift - 1) * r_x;
        }

        result *= checkpoints[Prefixes::Pow2W].unwrap_or(F::one());
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
        if suffix_len != 0 {
            return Some(F::one()).into();
        }

        // r_y is the highest bit of the shift amount
        if j == 2 * XLEN - 5 {
            let shift = 1 << 16; // 2^(32/2)
            return Some(F::one() + F::from_u64(shift - 1) * r_y).into();
        }

        // r_x and r_y are bits in the shift amount
        if 2 * XLEN - j < 5 {
            let mut checkpoint = checkpoints[Prefixes::Pow2W].unwrap();
            let shift = 1 << (1 << (2 * XLEN - j));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_x;
            let shift = 1 << (1 << (2 * XLEN - j - 1));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_y;
            return Some(checkpoint).into();
        }

        Some(F::one()).into()
    }
}
