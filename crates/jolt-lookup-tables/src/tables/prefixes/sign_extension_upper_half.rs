use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

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
}
