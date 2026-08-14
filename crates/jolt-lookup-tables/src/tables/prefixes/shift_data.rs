use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Joint accumulator `L_b·P_b` for the store-data shift tables:
/// `L` is the low `8·EIGHTHS` bits of the left operand (accumulated
/// additively across phases), and `P = 2^(8·(y mod 8 & (8−EIGHTHS)))` is the
/// offset scale, which multiplies in at the final phase (the offset bits of
/// `y` never straddle a phase boundary).
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
}

impl<F: Field, const EIGHTHS: usize> SparseDensePrefix<F> for ShiftDataPrefix<EIGHTHS> {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // The offset bits of y (interleaved index bits 0/2/4) stay in the
        // suffix until the final phase, and `x_lo` below assumes even phase
        // cuts; both pair with the OffsetScale suffix's `b.len() < 6` guard.
        debug_assert!(suffix_len == 0 || suffix_len >= 6);
        let (xb, yb) = b.uninterleave();
        let (x_val, y_val) = (u64::from(xb), u64::from(yb));
        // b covers index bits [suffix_len, suffix_len + b.len()); its x part
        // holds x bits starting at suffix_len/2 (phase cuts are even).
        let x_lo = suffix_len / 2;
        let mut lane = F::zero();
        for k in x_lo..Self::LANE_BITS.min(x_lo + xb.len()) {
            if (x_val >> (k - x_lo)) & 1 == 1 {
                lane += F::from_u64(1u64 << k);
            }
        }
        let value = checkpoints[Self::VARIANT] + lane;
        if suffix_len == 0 {
            let offset = y_val & Self::OFFSET_MASK;
            value * F::from_u64(1u64 << ((XLEN / 8) as u64 * offset))
        } else {
            value
        }
    }
}
