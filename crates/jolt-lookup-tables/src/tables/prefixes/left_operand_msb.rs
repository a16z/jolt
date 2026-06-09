use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

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

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        if j == 0 {
            F::from_u32(c)
        } else if j == 1 {
            r_x.unwrap().into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].unwrap()
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        if j == 1 {
            Some(r_x.into()).into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].into()
        }
    }
}
