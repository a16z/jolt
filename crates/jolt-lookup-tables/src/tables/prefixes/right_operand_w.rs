use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightOperandWPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandWPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let mut result = checkpoints[Prefixes::RightOperandW];
        if suffix_len < XLEN {
            let (_, y) = b.uninterleave();
            result += F::from_u128(u128::from(y) << (suffix_len / 2));
        }
        result
    }
}
