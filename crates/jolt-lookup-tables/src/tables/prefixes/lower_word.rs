use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LowerWordPrefix {}

impl<F: Field> SparseDensePrefix<F> for LowerWordPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start < XLEN {
            return F::zero();
        }
        checkpoints[Prefixes::LowerWord] + F::from_u128(u128::from(b) << suffix_len)
    }
}
