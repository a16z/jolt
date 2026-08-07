use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum SignExtensionWPrefix {}

impl<F: Field> SparseDensePrefix<F> for SignExtensionWPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        let sign_bit_round = XLEN;
        if j_start + b.len() <= sign_bit_round {
            return F::zero();
        }

        let raw = u128::from(b);
        let sign_bit = if j_start <= sign_bit_round {
            let offset = sign_bit_round - j_start;
            let bit = (raw >> (b.len() - 1 - offset)) & 1;
            F::from_u64(bit as u64)
        } else {
            checkpoints[Prefixes::WordMsb]
        };
        let mut result = if j_start <= sign_bit_round {
            sign_bit * F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)))
        } else {
            checkpoints[Prefixes::SignExtensionW]
        };

        for offset in 0..b.len() {
            let round = j_start + offset;
            if round <= sign_bit_round || round.is_multiple_of(2) {
                continue;
            }
            let pair = round / 2;
            if pair < XLEN / 2 {
                continue;
            }
            let position = pair - XLEN / 2;
            if position == 0 {
                continue;
            }
            let y_bit = (raw >> (b.len() - 1 - offset)) & 1;
            result += sign_bit * F::from_u128((1 - y_bit) << position);
        }

        result
    }
}
