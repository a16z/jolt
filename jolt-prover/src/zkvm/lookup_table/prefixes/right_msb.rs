use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RightMsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for RightMsbPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j == 0 {
            let y_msb = b.pop_msb();
            F::from_u8(y_msb)
        } else if j == 1 {
            F::from_u32(c)
        } else {
            checkpoints[Prefixes::RightOperandMsb].unwrap()
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j == 1 {
            Some(r_y.into()).into()
        } else {
            checkpoints[Prefixes::RightOperandMsb].into()
        }
    }
}
