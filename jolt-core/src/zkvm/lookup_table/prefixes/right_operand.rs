use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::read_raf_checking::current_suffix_len,
};

pub enum RightOperandPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for RightOperandPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut result = checkpoints[Prefixes::RightOperand].unwrap_or(F::zero());

        if j % 2 == 1 {
            // c is of the right operand
            let shift = XLEN - 1 - j / 2;
            result += F::from_u128((c as u128) << shift);
        }

        let (_x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(j);
        result += F::from_u128(u128::from(y) << (suffix_len / 2));

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: C,
        r_y: C,
        j: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let shift = XLEN - 1 - j / 2;
        let updated = checkpoints[Prefixes::RightOperand].unwrap_or(F::zero())
            + (F::from_u64(1 << shift) * r_y);
        Some(updated).into()
    }
}
