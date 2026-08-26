use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum WordMsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for WordMsbPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        _: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN {
            F::one()
        } else if j == XLEN {
            F::from_u32(c)
        } else if j == XLEN + 1 {
            r_x.unwrap().into()
        } else {
            checkpoints[Prefixes::WordMsb].unwrap()
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        _: C,
        j: usize,
        _: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j < XLEN + 1 {
            Some(F::one()).into()
        } else if j == XLEN + 1 {
            Some(r_x.into()).into()
        } else {
            checkpoints[Prefixes::WordMsb].into()
        }
    }
}
