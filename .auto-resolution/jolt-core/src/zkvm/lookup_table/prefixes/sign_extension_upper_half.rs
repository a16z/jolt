use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SignExtensionUpperHalfPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SignExtensionUpperHalfPrefix<XLEN> {
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
        let suffix_len = LOG_K - j - b.len() - 1;
        let half_word_size = XLEN / 2;

        // If suffix handles sign extension, return 1
        if suffix_len >= half_word_size {
            return F::one();
        }

        if j == XLEN + half_word_size {
            // Sign bit is in c
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)).mul_u64(c as u64)
        } else if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x.unwrap()
        } else if j > XLEN + half_word_size + 1 {
            // Sign bit has been processed, use checkpoint
            checkpoints[Prefixes::SignExtensionUpperHalf].unwrap_or(F::zero())
        } else {
            unreachable!("This case should never happen if our prefixes start at half_word_size");
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        _r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let half_word_size = XLEN / 2;

        if j == XLEN + half_word_size + 1 {
            // Sign bit is in r_x
            let value = F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x;
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionUpperHalf].into()
        }
    }
}
