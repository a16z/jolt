use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// `2^(32·ea_2)` where `ea_2` is bit 2 of the (non-interleaved) lookup index:
/// the word-offset half of a doubleword effective address.
pub enum Pow2OffsetWPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for Pow2OffsetWPrefix<XLEN> {
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
        // Phase-boundary-agnostic split of the index around bit 2: the suffix
        // owns bits [0, suffix_len), the unbound bits `b` own
        // [suffix_len, suffix_len + b.len()), the current variable `c` sits at
        // suffix_len + b.len(), and everything above is bound into the
        // checkpoint.
        let suffix_len = LOG_K - j - b.len() - 1;

        // Bit 2 is in the suffix; `Pow2OffsetWSuffix` supplies the factor.
        if suffix_len >= 3 {
            return F::one();
        }

        let checkpoint = checkpoints[Prefixes::Pow2OffsetW].unwrap_or(F::one());

        // Bit 2 is still among the unbound bits `b`.
        if suffix_len + b.len() > 2 {
            let bits: u128 = b.into();
            let bit2 = ((bits >> (2 - suffix_len)) & 1) as u32;
            return checkpoint * F::from_u64(1 << (32 * bit2));
        }

        // The current variable `c` is bit 2 of the index:
        // 2^(32·c) = 1 + (2^32 − 1)·c
        if suffix_len + b.len() == 2 {
            return checkpoint * (F::one() + F::from_u64((1u64 << 32) - 1) * F::from_u32(c));
        }

        // Bit 2 has been bound; its contribution lives in the checkpoint.
        checkpoint
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let checkpoint = checkpoints[Prefixes::Pow2OffsetW].unwrap_or(F::one());

        // `r_y` was bound at index bit LOG_K − 1 − j; bit 2 is even, so it is
        // always bound as an `r_y` (odd index bits are the `r_x` side).
        if LOG_K - 1 - j == 2 {
            let updated = checkpoint * (F::one() + F::from_u64((1u64 << 32) - 1) * r_y);
            return Some(updated).into();
        }

        Some(checkpoint).into()
    }
}
