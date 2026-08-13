use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// `P_b = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` over bound bits: 1 until the
/// final phase (the interleaved offset bits of `y` sit in the last index
/// bits), then the offset scale of the second operand. The offset bits are
/// all y bits, so the pending challenge `r_x` never contributes.
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
        // The offset bits of y (interleaved index bits 0/2/4) stay in the
        // suffix until the final phase; this pairing with the suffix's
        // `b.len() < 6` guard assumes phase boundaries never fall inside the
        // low six index bits.
        debug_assert!(suffix_len == 0 || suffix_len >= 6);
        if suffix_len != 0 {
            return F::one();
        }

        // Unbound offset bits of y are binary values in `b`; bound ones live
        // in the checkpoint.
        let (_, yb) = b.uninterleave();
        let offset = u64::from(yb) & Self::OFFSET_MASK;
        let mut value = checkpoints[Self::VARIANT].unwrap_or(F::one())
            * F::from_u64(1u64 << ((XLEN / 8) as u64 * offset));
        if j % 2 == 1 {
            // The current variable `c` is y bit m.
            let m = XLEN - 1 - (j - 1) / 2;
            if (Self::OFFSET_MASK >> m) & 1 == 1 {
                let scale = F::from_u128((1u128 << ((XLEN / 8) << m)) - 1);
                value *= F::one() + scale * F::from_u32(c);
            }
        }
        value
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
        debug_assert!(suffix_len == 0 || suffix_len >= 6);
        if suffix_len != 0 {
            return Some(F::one()).into();
        }

        let mut updated = checkpoints[Self::VARIANT].unwrap_or(F::one());
        // r_y is y bit m; fold in its scale factor if it is an offset bit.
        let m = XLEN - 1 - (j - 1) / 2;
        if m < 3 && (Self::OFFSET_MASK >> m) & 1 == 1 {
            let scale = F::from_u128((1u128 << ((XLEN / 8) << m)) - 1);
            updated *= F::one() + scale * r_y;
        }
        Some(updated).into()
    }
}
