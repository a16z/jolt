use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LeftMsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftMsbPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        _: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j == 0 {
            F::from_u32(c)
        } else if j == 1 {
            r_x.unwrap().into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].unwrap()
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        _: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j == 1 {
            Some(r_x.into()).into()
        } else {
            checkpoints[Prefixes::LeftOperandMsb].into()
        }
    }
}
