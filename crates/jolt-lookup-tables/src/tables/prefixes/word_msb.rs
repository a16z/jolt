use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum WordMsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for WordMsbPrefix {
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
