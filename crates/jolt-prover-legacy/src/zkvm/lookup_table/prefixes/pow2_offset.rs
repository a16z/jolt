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
        let suffix_len = LOG_K - j - b.len() - 1;
        // The low 3 bits of the raw index stay in the suffix until the final
        // phase; this pairing with the suffix's `b.len() < 3` guard assumes
        // phase boundaries never fall inside the low three index bits.
        debug_assert!(suffix_len == 0 || suffix_len >= 3);
        if suffix_len != 0 {
            return F::one();
        }

        let checkpoint = checkpoints[Self::VARIANT].unwrap_or(F::one());
        let bits: u128 = b.into();

        match b.len() {
            // Bits 2..0 are all still among the unbound bits `b`.
            n if n >= 3 => {
                let offset = (bits & Self::OFFSET_MASK) as u32;
                checkpoint * F::from_u64(1 << (8 * offset))
            }
            // The current variable `c` is bit 2; bits 1..0 are in `b`.
            2 => {
                let offset = (bits & (Self::OFFSET_MASK & 3)) as u32;
                checkpoint * Self::bit_factor(2, F::from_u32(c)) * F::from_u64(1 << (8 * offset))
            }
            // The current variable `c` is bit 1; bit 0 is in `b`; bit 2 has
            // been bound and lives in the checkpoint.
            1 => {
                let factor = if LOW_BIT <= 1 {
                    Self::bit_factor(1, F::from_u32(c))
                } else {
                    F::one()
                };
                let offset = (bits & (Self::OFFSET_MASK & 1)) as u32;
                checkpoint * factor * F::from_u64(1 << (8 * offset))
            }
            // The current variable `c` is bit 0; bit 1 is the pending
            // challenge `r_x`; bit 2 lives in the checkpoint. `r_x` is always
            // `Some` here: `b.len() == 0` with `suffix_len == 0` forces
            // `j = LOG_K − 1`, which is odd.
            _ => {
                let factor_1 = if LOW_BIT <= 1 {
                    Self::bit_factor(1, F::one() * r_x.unwrap())
                } else {
                    F::one()
                };
                let factor_0 = if LOW_BIT == 0 {
                    Self::bit_factor(0, F::from_u32(c))
                } else {
                    F::one()
                };
                checkpoint * factor_1 * factor_0
            }
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
        debug_assert!(suffix_len == 0 || suffix_len >= 3);
        if suffix_len != 0 {
            return Some(F::one()).into();
        }

        let checkpoint = checkpoints[Self::VARIANT].unwrap_or(F::one());

        // r_y is bit 2 of the index.
        if j == 2 * XLEN - 3 {
            let updated = checkpoint * Self::bit_factor(2, F::one() * r_y);
            return Some(updated).into();
        }

        // r_x is bit 1 and r_y is bit 0 of the index.
        if j == 2 * XLEN - 1 && LOW_BIT <= 1 {
            let mut updated = checkpoint * Self::bit_factor(1, F::one() * r_x);
            if LOW_BIT == 0 {
                updated *= Self::bit_factor(0, F::one() * r_y);
            }
            return Some(updated).into();
        }

        Some(checkpoint).into()
    }
}
