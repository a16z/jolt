use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Joint accumulator `L_b·P_b` for the store-data shift tables:
/// `L = Σ_k 2^k·x_k` over the low `EIGHTHS·(XLEN/8)` bits of the left operand
/// (x bits sit at odd index positions `2k+1`), and
/// `P = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` is the offset scale of the
/// second operand (offset bits y_0..y_2 sit at even index positions 0/2/4).
///
/// Phase-boundary-agnostic: each phase supplies the bits it owns, and the
/// suffix carries partial factors for the bits below the boundary.
///
/// Invariant: this prefix's checkpoint holds `L_bound·P_bound`, while the
/// matching `OffsetScale` prefix's checkpoint holds `P_bound`, so the
/// partially bound value is `(checkpoint + ΔL·P_bound)·ΔP`.
pub enum ShiftDataPrefix<const EIGHTHS: usize> {}

impl<const EIGHTHS: usize> ShiftDataPrefix<EIGHTHS> {
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

impl<F: JoltField, const EIGHTHS: usize> SparseDensePrefix<F> for ShiftDataPrefix<EIGHTHS> {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // The suffix owns index bits [0, suffix_len), this phase's bits `b`
        // own [suffix_len, suffix_len + b.len()), and everything above is
        // bound into the checkpoints.
        let bits: u128 = b.into();

        // ΔL: lane contribution of the x bits this phase owns.
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

        // (L_bound + ΔL)·P_bound = checkpoint + ΔL·(OffsetScale checkpoint)
        let offset_scale = checkpoints[Self::OFFSET_SCALE_VARIANT];
        let mut value = checkpoints[Self::VARIANT] + lane * offset_scale;

        // ΔP: offset factors of the y bits this phase owns.
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
        value
    }
}
