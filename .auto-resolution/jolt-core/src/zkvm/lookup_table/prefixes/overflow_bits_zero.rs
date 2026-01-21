use crate::zkvm::instruction_lookups::LOG_K;
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Returns 1 if the upper 128-XLEN bits are all zero (no overflow), 0 otherwise.
pub enum OverflowBitsZeroPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for OverflowBitsZeroPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let suffix_len = LOG_K - j - b.len() - 1;
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());
        }

        let mut result = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= (F::one() - r_x) * (F::one() - y);
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            result *= (F::one() - x) * (F::one() - y);
        }

        let rest = u128::from(b);
        let temp = F::from_u64((((rest << suffix_len) >> XLEN) == 0) as u64);
        result *= temp;

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].into();
        }
        let updated = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one())
            * (F::one() - r_x)
            * (F::one() - r_y);

        Some(updated).into()
    }
}
