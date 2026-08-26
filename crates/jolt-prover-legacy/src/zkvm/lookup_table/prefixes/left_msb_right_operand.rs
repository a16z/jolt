use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{
    left_msb::LeftMsbPrefix, right_operand::RightOperandPrefix, PrefixCheckpoint, SparseDensePrefix,
};

pub enum LeftMsbRightOperandPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LeftMsbRightOperandPrefix<XLEN> {
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
            * RightOperandPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
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
        let right = RightOperandPrefix::<XLEN>::update_prefix_checkpoint(
            checkpoints,
            r_x,
            r_y,
            j,
            suffix_len,
        )
        .0
        .unwrap_or(F::zero());
        Some(left * right).into()
    }
}
