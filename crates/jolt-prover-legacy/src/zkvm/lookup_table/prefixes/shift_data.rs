use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Joint accumulator `L_b·P_b` for the store-data shift tables:
/// `L = Σ_k 2^k·x_k` over the low `EIGHTHS·(XLEN/8)` bits of the left operand
/// (x bits sit at odd index positions `2k+1`), and
/// `P = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` is the offset scale of the
/// second operand (offset bits y_0..y_2 sit at even index positions 0/2/4).
///
/// Phase-boundary-agnostic: each call owns the bits in `b` plus the current
/// variable `c` and pending challenge `r_x`; the suffix supplies partial
/// factors for the bits it owns.
///
/// Invariant: this prefix's checkpoint holds `L_bound·P_bound`, while the
/// matching `OffsetScale` prefix's checkpoint holds `P_bound`, so the
/// partially bound value is `(checkpoint + ΔL·P_bound)·ΔP`.
pub enum ShiftDataPrefix<const XLEN: usize, const EIGHTHS: usize> {}

impl<const XLEN: usize, const EIGHTHS: usize> ShiftDataPrefix<XLEN, EIGHTHS> {
    const LANE_BITS: usize = EIGHTHS * (XLEN / 8);
    // `8 − EIGHTHS` clears the low log2(EIGHTHS) offset bits only because
    // EIGHTHS ∈ {1, 2, 4} is a power of two.
    const OFFSET_MASK: u64 = (8 - EIGHTHS) as u64;

    const VARIANT: Prefixes = match EIGHTHS {
        1 => Prefixes::ShiftDataB,
        2 => Prefixes::ShiftDataH,
        4 => Prefixes::ShiftDataW,
        _ => panic!("unsupported EIGHTHS"),
    };

    const OFFSET_SCALE_VARIANT: Prefixes = match EIGHTHS {
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
    for ShiftDataPrefix<XLEN, EIGHTHS>
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
        // The suffix owns index bits [0, suffix_len), the unbound bits `b`
        // own [suffix_len, suffix_len + b.len()), the current variable `c`
        // sits at suffix_len + b.len(), the pending challenge `r_x` (when
        // present) one above that, and everything higher is bound into the
        // checkpoints.
        let suffix_len = LOG_K - j - b.len() - 1;
        let bits: u128 = b.into();

        // ΔL: lane contribution of the x bits this call owns.
        let mut lane = F::zero();
        for k in 0..Self::LANE_BITS {
            let pos = 2 * k + 1;
            if pos >= suffix_len
                && pos < suffix_len + b.len()
                && (bits >> (pos - suffix_len)) & 1 == 1
            {
                lane += F::from_u64(1u64 << k);
            }
        }
        let c_pos = suffix_len + b.len();
        if !c_pos.is_multiple_of(2) {
            // The current variable `c` is x bit k.
            let k = (c_pos - 1) / 2;
            if k < Self::LANE_BITS {
                lane += F::from_u64(1u64 << k) * F::from_u32(c);
            }
        } else if let Some(r_x) = r_x {
            // The pending challenge `r_x` is x bit k (position c_pos + 1).
            let k = c_pos / 2;
            if k < Self::LANE_BITS {
                lane += r_x * F::from_u64(1u64 << k);
            }
        }

        // (L_bound + ΔL)·P_bound = checkpoint + ΔL·(OffsetScale checkpoint)
        let offset_scale = checkpoints[Self::OFFSET_SCALE_VARIANT].unwrap_or(F::one());
        let mut value = checkpoints[Self::VARIANT].unwrap_or(F::zero()) + lane * offset_scale;

        // ΔP: offset factors of the y bits this call owns.
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
        if c_pos.is_multiple_of(2) {
            // The current variable `c` is y bit i. `r_x` sits at an odd
            // position, so it is never an offset bit.
            let i = c_pos / 2;
            if i < 3 && (Self::OFFSET_MASK >> i) & 1 == 1 {
                value *= Self::bit_factor(i, F::from_u32(c));
            }
        }
        value
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
        let mut updated = checkpoints[Self::VARIANT].unwrap_or(F::zero());
        // r_x is x bit k; a lane bit's contribution is added scaled by the
        // offset-scale of the previously bound offset bits.
        let k = XLEN - 1 - (j - 1) / 2;
        if k < Self::LANE_BITS {
            let offset_scale = checkpoints[Self::OFFSET_SCALE_VARIANT].unwrap_or(F::one());
            updated += (r_x * F::from_u64(1u64 << k)) * offset_scale;
        }
        // r_y is y bit i (index position LOG_K − 1 − j); an offset bit's
        // scale factor folds in multiplicatively. The add-then-multiply order
        // keeps the `L_bound·P_bound` invariant when a pair binds both a lane
        // bit and an offset bit.
        let i = (LOG_K - 1 - j) / 2;
        if i < 3 && (Self::OFFSET_MASK >> i) & 1 == 1 {
            updated *= Self::bit_factor(i, F::one() * r_y);
        }
        Some(updated).into()
    }
}
