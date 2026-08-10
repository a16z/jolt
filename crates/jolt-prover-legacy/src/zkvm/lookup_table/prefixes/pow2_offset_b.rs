use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// `2^(8·(ea mod 8))` where `ea` is the (non-interleaved) lookup index: the
/// byte-offset scale of a doubleword effective address.
///
/// The offset is `4·ea_2 + 2·ea_1 + ea_0`, so the prefix is the product of
/// per-bit factors `2^(8·2^k·ea_k) = 1 + (2^(8·2^k) − 1)·ea_k` for
/// `k = 2, 1, 0`.
pub enum Pow2OffsetBPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for Pow2OffsetBPrefix<XLEN> {
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
        // The low 3 bits of the raw index stay in the suffix until the final phase.
        if suffix_len != 0 {
            return F::one();
        }

        let checkpoint = checkpoints[Prefixes::Pow2OffsetB].unwrap_or(F::one());

        // Bits 2..0 are all still among the unbound bits `b`.
        if b.len() >= 3 {
            let bits: u128 = b.into();
            let offset = (bits & 7) as u32;
            return checkpoint * F::from_u64(1 << (8 * offset));
        }

        // The current variable `c` is bit 2; bits 1..0 are in `b`:
        // 2^(32·c) = 1 + (2^32 − 1)·c
        if b.len() == 2 {
            let bits: u128 = b.into();
            let offset = (bits & 3) as u32;
            return checkpoint
                * (F::one() + F::from_u64((1u64 << 32) - 1) * F::from_u32(c))
                * F::from_u64(1 << (8 * offset));
        }

        // The current variable `c` is bit 1; bit 0 is in `b`; bit 2 has been
        // bound and lives in the checkpoint.
        if b.len() == 1 {
            let bits: u128 = b.into();
            let offset = (bits & 1) as u32;
            return checkpoint
                * (F::one() + F::from_u64((1u64 << 16) - 1) * F::from_u32(c))
                * F::from_u64(1 << (8 * offset));
        }

        // The current variable `c` is bit 0; bit 1 is the pending challenge
        // `r_x`; bit 2 lives in the checkpoint.
        checkpoint
            * (F::one() + F::from_u64((1u64 << 16) - 1) * r_x.unwrap())
            * (F::one() + F::from_u64((1u64 << 8) - 1) * F::from_u32(c))
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
        if suffix_len != 0 {
            return Some(F::one()).into();
        }

        let checkpoint = checkpoints[Prefixes::Pow2OffsetB].unwrap_or(F::one());

        // r_y is bit 2 of the index
        if j == 2 * XLEN - 3 {
            let updated = checkpoint * (F::one() + F::from_u64((1u64 << 32) - 1) * r_y);
            return Some(updated).into();
        }

        // r_x is bit 1 and r_y is bit 0 of the index
        if j == 2 * XLEN - 1 {
            let updated = checkpoint
                * (F::one() + F::from_u64((1u64 << 16) - 1) * r_x)
                * (F::one() + F::from_u64((1u64 << 8) - 1) * r_y);
            return Some(updated).into();
        }

        Some(checkpoint).into()
    }
}
