use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum LsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: Field> SparseDensePrefix<F> for LsbPrefix<XLEN> {
    fn prefix_mle<C>(
        _checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let suffix_len = 2 * XLEN - j - b.len() - 1;
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

    fn update_prefix_checkpoint<C>(
        _: &[PrefixCheckpoint<F>],
        _: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j == 2 * XLEN - 1 {
            Some(r_y.into()).into()
        } else {
            Some(F::one()).into()
        }
    }
}
