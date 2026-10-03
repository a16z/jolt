use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum SignExtensionUpperHalfPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for SignExtensionUpperHalfPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let half_word_size = XLEN / 2;

        if suffix_len >= half_word_size {
            return F::one();
        }

        let j_start = 2 * XLEN - suffix_len - b.len();
        let sign_bit_round = XLEN + half_word_size;

        if j_start <= sign_bit_round && sign_bit_round < j_start + b.len() {
            let (x, _y) = b.uninterleave();
            let x_val = u64::from(x);
            let sign_bit = (x_val >> (x.len() - 1)) & 1;
            F::from_u128(((1u128 << half_word_size) - 1) << half_word_size) * F::from_u64(sign_bit)
        } else {
            checkpoints[Prefixes::SignExtensionUpperHalf]
        }
    }
}
