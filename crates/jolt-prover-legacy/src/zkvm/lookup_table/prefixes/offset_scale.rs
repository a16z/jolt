use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// `P_b = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` over the bits the prefix
/// owns: the offset scale of the second operand. The offset bits y_0..y_2 sit
/// at even index positions 0/2/4, so the pending challenge `r_x` (always at
/// an odd position) never contributes.
///
/// Phase-boundary-agnostic: the scale is a product of per-bit factors
/// `1 + (2^((XLEN/8)·2^i) − 1)·y_i`, so each side supplies exactly the bits
/// it owns (the suffix carries partial factors for offset bits below the
/// boundary).
pub enum OffsetScalePrefix<const XLEN: usize, const EIGHTHS: usize> {}

impl<const XLEN: usize, const EIGHTHS: usize> OffsetScalePrefix<XLEN, EIGHTHS> {
    // `8 − EIGHTHS` clears the low log2(EIGHTHS) offset bits only because
    // EIGHTHS ∈ {1, 2, 4} is a power of two.
    const OFFSET_MASK: u64 = (8 - EIGHTHS) as u64;

    const VARIANT: Prefixes = match EIGHTHS {
        1 => Prefixes::OffsetScaleB,
        2 => Prefixes::OffsetScaleH,
        4 => Prefixes::OffsetScaleW,
        _ => panic!("unsupported EIGHTHS"),
    };

    /// `1 + (2^((XLEN/8)·2^i) − 1)·y_i`, the multilinear per-offset-bit factor.
    fn bit_factor<F: JoltField>(i: usize, bit: F) -> F {
        F::one() + F::from_u128((1u128 << ((XLEN / 8) << i)) - 1) * bit
    }
}

impl<const XLEN: usize, const EIGHTHS: usize, F: JoltField> SparseDensePrefix<F>
    for OffsetScalePrefix<XLEN, EIGHTHS>
{
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
        if suffix_len > 4 {
            // All offset bits are in the suffix, which supplies the factor.
            return F::one();
        }

        let bits: u128 = b.into();
        let mut value = checkpoints[Self::VARIANT].unwrap_or(F::one());
        for i in 0..3 {
            if (Self::OFFSET_MASK >> i) & 1 == 0 {
                continue;
            }
            let pos = 2 * i;
            if pos >= suffix_len
                && pos < suffix_len + b.len()
                && (bits >> (pos - suffix_len)) & 1 == 1
            {
                value *= F::from_u128(1u128 << ((XLEN / 8) << i));
            }
        }
        let c_pos = suffix_len + b.len();
        if c_pos.is_multiple_of(2) {
            // The current variable `c` is y bit i.
            let i = c_pos / 2;
            if i < 3 && (Self::OFFSET_MASK >> i) & 1 == 1 {
                value *= Self::bit_factor(i, F::from_u32(c));
            }
        }
        value
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
        let mut updated = checkpoints[Self::VARIANT].unwrap_or(F::one());
        // r_y is y bit i (index position LOG_K − 1 − j); fold in its scale
        // factor if it is an offset bit. r_x sits at an odd position and is
        // never an offset bit.
        let i = (LOG_K - 1 - j) / 2;
        if i < 3 && (Self::OFFSET_MASK >> i) & 1 == 1 {
            updated *= Self::bit_factor(i, F::one() * r_y);
        }
        Some(updated).into()
    }
}
