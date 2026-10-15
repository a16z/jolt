use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Prefix-owned portion of SRLW's sign-extension predicate `x_{w-1} * y_0`.
///
/// Here `w = XLEN / 2` and `y = 2^w - 2^s`, so `y_0` is one exactly when
/// `s = 0`. A logical word shift can therefore leave result bit `w - 1` set
/// only when `x_{w-1} * y_0 = 1`. The checkpoint is zero until `x_{w-1}`
/// enters the prefix, carries `x_{w-1}` while `y_0` remains in the suffix, and
/// folds in `y_0` during the final phase. `X31Y0Suffix` supplies the
/// same product while both bits belong to the suffix.
pub enum SrlwSextPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for SrlwSextPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        let sign_bit_round = XLEN;

        let sign_bit = if j_start <= sign_bit_round && sign_bit_round < j_start + b.len() {
            let offset = sign_bit_round - j_start;
            let bit = (u128::from(b) >> (b.len() - 1 - offset)) & 1;
            F::from_u64(bit as u64)
        } else if j_start > sign_bit_round {
            checkpoints[Prefixes::SrlwSext]
        } else {
            F::zero()
        };

        if suffix_len == 0 {
            sign_bit * F::from_u64((u128::from(b) & 1) as u64)
        } else {
            sign_bit
        }
    }
}
