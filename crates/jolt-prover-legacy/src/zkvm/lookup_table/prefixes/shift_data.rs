use crate::field::{ChallengeFieldOps, FieldChallengeOps};
use crate::zkvm::instruction_lookups::LOG_K;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Joint accumulator `L_b·P_b` for the store-data shift tables:
/// `L` is the low `EIGHTHS·(XLEN/8)` bits of the left operand (accumulated
/// additively across phases), and `P = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))`
/// is the offset scale, which multiplies in during the final phase (the
/// offset bits of `y` never straddle a phase boundary).
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
        let suffix_len = LOG_K - j - b.len() - 1;
        // The offset bits of y (interleaved index bits 0/2/4) stay in the
        // suffix until the final phase, and `x_lo` below assumes even phase
        // cuts; both pair with the OffsetScale suffix's `b.len() < 6` guard.
        debug_assert!(suffix_len == 0 || suffix_len >= 6);
        let (xb, yb) = b.uninterleave();
        let x_val = u64::from(xb);

        // Lane contribution of this phase's x bits: the unbound bits in `b`
        // (whose x part starts at absolute x bit suffix_len/2; phase cuts are
        // even) plus the current variable / pending challenge.
        let x_lo = suffix_len / 2;
        let mut lane = F::zero();
        for k in x_lo..Self::LANE_BITS.min(x_lo + xb.len()) {
            if (x_val >> (k - x_lo)) & 1 == 1 {
                lane += F::from_u64(1u64 << k);
            }
        }
        if j.is_multiple_of(2) {
            // The current variable `c` is x bit m.
            let m = XLEN - 1 - j / 2;
            if m < Self::LANE_BITS {
                lane += F::from_u64(1u64 << m) * F::from_u32(c);
            }
        } else if let Some(r_x) = r_x {
            // The pending challenge `r_x` is x bit m.
            let m = XLEN - 1 - (j - 1) / 2;
            if m < Self::LANE_BITS {
                lane += r_x * F::from_u64(1u64 << m);
            }
        }

        // (L_bound + ΔL)·P_bound = checkpoint + ΔL·(OffsetScale checkpoint)
        let offset_scale = checkpoints[Self::OFFSET_SCALE_VARIANT].unwrap_or(F::one());
        let mut value = checkpoints[Self::VARIANT].unwrap_or(F::zero()) + lane * offset_scale;

        // In the final phase, fold in the offset scale of the unbound and
        // current offset bits of y (bound ones live in the checkpoints).
        if suffix_len == 0 {
            let offset = u64::from(yb) & Self::OFFSET_MASK;
            value *= F::from_u64(1u64 << ((XLEN / 8) as u64 * offset));
            if j % 2 == 1 {
                // The current variable `c` is y bit m.
                let m = XLEN - 1 - (j - 1) / 2;
                if (Self::OFFSET_MASK >> m) & 1 == 1 {
                    let scale = F::from_u128((1u128 << ((XLEN / 8) << m)) - 1);
                    value *= F::one() + scale * F::from_u32(c);
                }
            }
        }
        value
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
        debug_assert!(suffix_len == 0 || suffix_len >= 6);
        let mut updated = checkpoints[Self::VARIANT].unwrap_or(F::zero());
        // r_x is x bit m; a lane bit's contribution is added scaled by the
        // offset-scale of the previously bound offset bits (1 until the
        // final phase).
        let m = XLEN - 1 - (j - 1) / 2;
        if m < Self::LANE_BITS {
            let offset_scale = checkpoints[Self::OFFSET_SCALE_VARIANT].unwrap_or(F::one());
            updated += (r_x * F::from_u64(1u64 << m)) * offset_scale;
        }
        // r_y is y bit m; an offset bit's scale factor folds in
        // multiplicatively (offset bits only appear in the final phase).
        if suffix_len == 0 && m < 3 && (Self::OFFSET_MASK >> m) & 1 == 1 {
            let scale = F::from_u128((1u128 << ((XLEN / 8) << m)) - 1);
            updated *= F::one() + scale * r_y;
        }
        Some(updated).into()
    }
}
