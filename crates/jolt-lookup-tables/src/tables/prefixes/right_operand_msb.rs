use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightOperandMsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandMsbPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start > 0 {
            return checkpoints[Prefixes::RightOperandMsb];
        }
        let (_, y) = b.uninterleave();
        F::from_u64(u64::from(y) >> (y.len() - 1))
    }

    #[expect(clippy::unwrap_used)]
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        if j == 0 {
            let y_msb = b.pop_msb();
            F::from_u8(y_msb)
        } else if j == 1 {
            F::from_u32(c)
        } else {
            checkpoints[Prefixes::RightOperandMsb].unwrap()
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
            Some(r_y).into()
        } else {
            checkpoints[Prefixes::RightOperandMsb].into()
        }
    }
}
