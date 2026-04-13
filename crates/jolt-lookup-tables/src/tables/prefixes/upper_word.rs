use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum UpperWordPrefix {}

impl<F: Field> SparseDensePrefix<F> for UpperWordPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start >= XLEN {
            return checkpoints[Prefixes::UpperWord];
        }

        let mut result = checkpoints[Prefixes::UpperWord];
        if suffix_len > XLEN {
            result += F::from_u64(u64::from(b) << (suffix_len - XLEN));
        } else {
            let (b_high, _) = b.split(XLEN - suffix_len);
            result += F::from_u64(u64::from(b_high));
        }
        result
    }
}
