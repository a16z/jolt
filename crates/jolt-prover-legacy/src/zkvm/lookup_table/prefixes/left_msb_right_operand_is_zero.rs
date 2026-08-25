use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{
    left_msb::LeftMsbPrefix, right_is_zero::RightOperandIsZeroPrefix, PrefixCheckpoint,
    SparseDensePrefix,
};

pub enum LeftMsbRightOperandIsZeroPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftMsbRightOperandIsZeroPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        LeftMsbPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            * RightOperandIsZeroPrefix::prefix_mle(checkpoints, r_x, c, b, j)
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let left = LeftMsbPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            .0
            .unwrap_or(F::zero());
        let right_is_zero = RightOperandIsZeroPrefix::update_prefix_checkpoint(
            checkpoints,
            r_x,
            r_y,
            j,
            suffix_len,
        )
        .0
        .unwrap_or(F::one());
        Some(left * right_is_zero).into()
    }
}
