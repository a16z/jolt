use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum Pow2Prefix {}

impl<F: Field> SparseDensePrefix<F> for Pow2Prefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // pow2 computes 2^(shift) where shift is the last log2(XLEN) interleaved
        // index bits. When the suffix still contains all shift bits, the prefix is 1.
        if suffix_len != 0 {
            return F::one();
        }

        // At suffix_len == 0, the shift bits are in the low bits of `b` (raw interleaved).
        // `b & (XLEN-1)` extracts the last 6 bits.
        checkpoints[Prefixes::Pow2] * F::from_u64(1u64 << (b & (crate::XLEN - 1)))
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        // Calculate suffix_len: LOG_K - j (current index) - b.len() (b index) - 1 (c bit)
        let suffix_len = LOG_K - j - b.len() - 1;
        if suffix_len != 0 {
            // Handled by suffix
            return F::one();
        }

        // Shift amount is the last XLEN bits of b
        if b.len() >= (XLEN.ilog2() as usize) {
            return F::from_u64(1 << (b & (XLEN - 1)));
        }

        let mut result = F::from_u64(1 << (b & (XLEN - 1)));
        let mut num_bits = b.len();
        let mut shift = 1u64 << (1u64 << num_bits);
        result *= F::from_u64(1 + (shift - 1) * c as u64);

        // Shift amount is [c, b]
        if b.len() == (XLEN.ilog2() as usize) - 1 {
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
        if j == 2 * XLEN - (XLEN.ilog2() as usize) {
            let shift = 1 << (XLEN / 2);
            return Some(F::one() + F::from_u64(shift - 1) * r_y).into();
        }

        // r_x and r_y are bits in the shift amount
        if 2 * XLEN - j < (XLEN.ilog2() as usize) {
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
