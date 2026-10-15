use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Prefix-owned portion of the SRAW sign-fill term.
///
/// Let `w = XLEN / 2`, `a = x_{w-1}`, and let `y` be the bitmask
/// `2^w - 2^s`. The complete sign-fill contribution is
///
/// `a * ((2^XLEN - 2^w) + sum_{i=1}^{w-1} 2^i * (1 - y_{w-1-i}))`.
///
/// The first term fills the upper word; the sum fills the `s` vacated bits in
/// the shifted word. This checkpoint starts at zero, adds the upper-word term
/// when `a` enters the prefix, then adds each mask-dependent term as its `y`
/// bit enters the prefix. `SignExtensionWSuffix` owns the terms that
/// still depend on suffix bits.
pub enum SignExtensionWPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for SignExtensionWPrefix {
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
