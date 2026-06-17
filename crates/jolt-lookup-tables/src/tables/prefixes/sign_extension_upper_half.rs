use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix, LOG_K};

pub enum SignExtensionUpperHalfPrefix {}

impl<F: Field> SparseDensePrefix<F> for SignExtensionUpperHalfPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let half_word_size = XLEN / 2;

        // Only defined on the lower half-word; returns 1 for higher bits.
        if suffix_len >= half_word_size {
            return F::one();
        }

        // The value is: sign_bit * ((2^half_word_size - 1) << half_word_size)
        // where sign_bit is the MSB of the lower half-word's x operand.
        // This is captured in the checkpoint after the first round where it's relevant.
        // At binary points the sign_bit is either from the current phase or the checkpoint.
        //
        // j_start = 2*XLEN - suffix_len - b.len()
        // The sign bit round is at j = XLEN + half_word_size (the first x bit of lower half).
        let j_start = 2 * XLEN - suffix_len - b.len();
        let sign_bit_round = XLEN + half_word_size;

        if j_start <= sign_bit_round && sign_bit_round < j_start + b.len() {
            // Sign bit is in this phase's b bits
            let (x, _y) = b.uninterleave();
            let x_val = u64::from(x);
            // The sign bit is the MSB of x in this phase
            let sign_bit = (x_val >> (x.len() - 1)) & 1;
            F::from_u128(((1u128 << half_word_size) - 1) << half_word_size) * F::from_u64(sign_bit)
        } else {
            checkpoints[Prefixes::SignExtensionUpperHalf]
        }
    }

    #[expect(clippy::unwrap_used)]
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        let half_word_size = XLEN / 2;

        // If suffix handles sign extension, return 1
        if suffix_len >= half_word_size {
            return F::one();
        }

        if j == XLEN + half_word_size {
            // Sign bit is in c
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)).mul_u64(c as u64)
        } else if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x.unwrap()
        } else if j > XLEN + half_word_size + 1 {
            // Sign bit has been processed, use checkpoint
            checkpoints[Prefixes::SignExtensionUpperHalf].unwrap_or(F::zero())
        } else {
            unreachable!("This case should never happen if our prefixes start at half_word_size");
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        let half_word_size = XLEN / 2;

        if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            let value = F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x;
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionUpperHalf].into()
        }
    }
}
