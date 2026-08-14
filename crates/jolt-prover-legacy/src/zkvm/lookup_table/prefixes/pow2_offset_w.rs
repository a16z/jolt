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
        let suffix_len = LOG_K - j - b.len() - 1;
        // Bit 2 of the raw index stays in the suffix until the final phase;
        // this pairing with the suffix's `b.len() < 3` guard assumes phase
        // boundaries never fall inside the low three index bits.
        debug_assert!(suffix_len == 0 || suffix_len >= 3);
        if suffix_len != 0 {
            return F::one();
        }

        // Bit 2 is still among the unbound bits `b`.
        if b.len() >= 3 {
            let bits: u128 = b.into();
            let bit2 = ((bits >> 2) & 1) as u32;
            return checkpoints[Prefixes::Pow2OffsetW].unwrap_or(F::one())
                * F::from_u64(1 << (32 * bit2));
        }

        // The current variable `c` is bit 2 of the index:
        // 2^(32·c) = 1 + (2^32 − 1)·c
        if b.len() == 2 {
            return checkpoints[Prefixes::Pow2OffsetW].unwrap_or(F::one())
                * (F::one() + F::from_u64((1u64 << 32) - 1) * F::from_u32(c));
        }

        // Bit 2 has been bound; its contribution lives in the checkpoint.
        checkpoints[Prefixes::Pow2OffsetW].unwrap_or(F::one())
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        debug_assert!(suffix_len == 0 || suffix_len >= 3);
        if suffix_len != 0 {
            return Some(F::one()).into();
        }

        // r_y is bit 2 of the index
        if j == 2 * XLEN - 3 {
            let updated = checkpoints[Prefixes::Pow2OffsetW].unwrap_or(F::one())
                * (F::one() + F::from_u64((1u64 << 32) - 1) * r_y);
            return Some(updated).into();
        }

        Some(checkpoints[Prefixes::Pow2OffsetW].unwrap_or(F::one())).into()
    }
}
