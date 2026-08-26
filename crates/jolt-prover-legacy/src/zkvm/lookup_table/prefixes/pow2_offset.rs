use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// `2^(8·offset)` where `offset` is the low 3 bits of the (non-interleaved)
/// lookup index with bits below `LOW_BIT` zeroed: the lane-offset scale of a
/// doubleword effective address. `LOW_BIT = 2` is the word variant (bit 2
/// only), `1` the halfword variant (bits 2-1), `0` the byte variant (bits
/// 2-0).
///
/// The offset is `Σ_{k=LOW_BIT..=2} 2^k·ea_k`, so the prefix is the product
/// of per-bit factors `2^(8·2^k·ea_k) = 1 + (2^(8·2^k) − 1)·ea_k`.
pub enum Pow2OffsetPrefix<const XLEN: usize, const LOW_BIT: usize> {}

impl<const XLEN: usize, const LOW_BIT: usize> Pow2OffsetPrefix<XLEN, LOW_BIT> {
    /// Which of the low 3 index bits contribute to the offset.
    const OFFSET_MASK: u128 = ((7 >> LOW_BIT) << LOW_BIT) as u128;

    const VARIANT: Prefixes = match LOW_BIT {
        0 => Prefixes::Pow2OffsetB,
        1 => Prefixes::Pow2OffsetH,
        2 => Prefixes::Pow2OffsetW,
        _ => panic!("unsupported LOW_BIT"),
    };

    /// `1 + (2^(8·2^k) − 1)·bit_k`, the multilinear per-bit factor.
    fn bit_factor<F: JoltField>(k: usize, bit: F) -> F {
        F::one() + F::from_u64((1u64 << (8 << k)) - 1) * bit
    }
}

impl<const XLEN: usize, const LOW_BIT: usize, F: JoltField> SparseDensePrefix<F>
    for Pow2OffsetPrefix<XLEN, LOW_BIT>
{
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
        // The 8-bit lane granularity below is what makes this the doubleword
        // lane offset; other instantiations are a compile error.
        const { assert!(XLEN == 64, "Pow2Offset hardcodes 8-bit lanes") };
        // Phase-boundary-agnostic split of the index around the low 3 offset
        // bits: the suffix owns bits [0, suffix_len), the unbound bits `b`
        // own [suffix_len, suffix_len + b.len()), the current variable `c`
        // sits at suffix_len + b.len(), the pending challenge `r_x` (when
        // present) one above that, and everything higher is bound into the
        // checkpoint. Each side supplies the per-bit factors it owns (the
        // suffixes carry partial factors for offset bits below a boundary).
        let suffix_len = LOG_K - j - b.len() - 1;
        if suffix_len >= 3 {
            // All offset bits are in the suffix, which supplies the factor.
            return F::one();
        }

        let checkpoint = checkpoints[Self::VARIANT].unwrap_or(F::one());
        let bits: u128 = b.into();
        let offset_in_b = ((bits << suffix_len) & Self::OFFSET_MASK) as u32;
        let mut result = checkpoint * F::from_u64(1 << (8 * offset_in_b));

        let c_pos = suffix_len + b.len();
        if (LOW_BIT..3).contains(&c_pos) {
            result *= Self::bit_factor(c_pos, F::from_u32(c));
        }
        // Offset bits 0 and 2 sit on the even (y) side of the interleaving,
        // so bit 1 is the only offset bit a pending `r_x` can be. `r_x` is
        // always `Some` here: `c` at bit 0 means `j = LOG_K − 1`, which is
        // odd (mid-pair).
        if c_pos == 0 && LOW_BIT <= 1 {
            result *= Self::bit_factor(1, F::one() * r_x.unwrap());
        }
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
        let checkpoint = checkpoints[Self::VARIANT].unwrap_or(F::one());

        // The pair binds `r_x` at index bit LOG_K − j and `r_y` at
        // LOG_K − 1 − j; keying on the bound bit's position (rather than the
        // final-phase round number) folds each offset bit into the checkpoint
        // exactly once under any phase layout.
        let y_pos = LOG_K - 1 - j;

        // r_y is bit 2 of the index.
        if y_pos == 2 {
            let updated = checkpoint * Self::bit_factor(2, F::one() * r_y);
            return Some(updated).into();
        }

        // r_x is bit 1 and r_y is bit 0 of the index.
        if y_pos == 0 {
            let mut updated = checkpoint;
            if LOW_BIT <= 1 {
                updated *= Self::bit_factor(1, F::one() * r_x);
            }
            if LOW_BIT == 0 {
                updated *= Self::bit_factor(0, F::one() * r_y);
            }
            return Some(updated).into();
        }

        Some(checkpoint).into()
    }
}
