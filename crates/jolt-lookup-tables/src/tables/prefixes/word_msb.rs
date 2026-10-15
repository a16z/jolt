use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Prefix checkpoint for the low word's most-significant bit.
///
/// Let `w = XLEN / 2`. At a Boolean point this checkpoint is `x_{w-1}`
/// (`x_31` for RV64). It remains the multiplicative identity while that bit
/// belongs to the suffix. Once a phase binds the bit, the checkpoint carries
/// its value so `VirtualSRAWTable::combine` can multiply it by the remaining
/// suffix-owned sign-fill terms.
pub enum WordMsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for WordMsbPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        let sign_bit_round = XLEN;

        if j_start + b.len() <= sign_bit_round {
            return F::one();
        }
        if j_start > sign_bit_round {
            return checkpoints[Prefixes::WordMsb];
        }

        let offset = sign_bit_round - j_start;
        let sign_bit = (u128::from(b) >> (b.len() - 1 - offset)) & 1;
        F::from_u64(sign_bit as u64)
    }
}
