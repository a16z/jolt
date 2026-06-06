use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, SparseDensePrefix, LOG_K};

pub enum LsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for LsbPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(_checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len == 0 {
            F::from_u64(u64::from(b) & 1)
        } else {
            F::one()
        }
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        let suffix_len = LOG_K - j - b.len() - 1;
        if j == 2 * XLEN - 1 {
            // in the log(K)th round, `c` corresponds to the LSB
            debug_assert_eq!(b.len(), 0);
            F::from_u32(c)
        } else if suffix_len == 0 {
            // in the (log(K)-1)th round, the LSB of `b` is the LSB
            F::from_u32(u32::from(b) & 1)
        } else {
            F::one()
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
        if j == 2 * XLEN - 1 {
            Some(r_y.into()).into()
        } else {
            Some(F::one()).into()
        }
    }
}
