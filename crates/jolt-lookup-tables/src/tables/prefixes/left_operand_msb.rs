use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftOperandMsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftOperandMsbPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start > 0 {
            return checkpoints[Prefixes::LeftOperandMsb];
        }
        // Phase 0: MSB of x is the MSB of the interleaved bits
        let (x, _) = b.uninterleave();
        F::from_u64(u64::from(x) >> (x.len() - 1))
    }
}
