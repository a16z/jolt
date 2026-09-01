use crate::zkvm::instruction_lookups::LOG_K;
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SrlwSextPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SrlwSextPrefix<XLEN> {
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
        let sign_bit = if j < XLEN {
            F::zero()
        } else if j == XLEN {
            F::from_u32(c)
        } else if j == XLEN + 1 {
            r_x.unwrap().into()
        } else {
            checkpoints[Prefixes::SrlwSext].unwrap()
        };

        let suffix_len = LOG_K - j - b.len() - 1;
        if suffix_len != 0 {
            return sign_bit;
        }
        if j == 2 * XLEN - 1 {
            sign_bit * F::from_u32(c)
        } else {
            sign_bit * F::from_u64((u128::from(b) & 1) as u64)
        }
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
        if j < XLEN + 1 {
            return Some(F::zero()).into();
        }
        if j == XLEN + 1 {
            return Some(r_x.into()).into();
        }
        let sign_bit = checkpoints[Prefixes::SrlwSext].unwrap();
        if suffix_len == 0 && j == 2 * XLEN - 1 {
            Some(sign_bit * r_y).into()
        } else {
            Some(sign_bit).into()
        }
    }
}
