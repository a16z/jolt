use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `P_b = 2^(8·(y mod 8 & (8−EIGHTHS)))` over bound bits: 1 until the final
/// phase (the interleaved offset bits of `y` sit in the last 5 index bits),
/// then the offset scale of the second operand.
///
/// Raw-index counterpart: [`super::pow2_offset::Pow2OffsetPrefix`] (same
/// three scale values over a single-operand index).
pub enum OffsetScalePrefix<const EIGHTHS: usize> {}

impl<const EIGHTHS: usize> OffsetScalePrefix<EIGHTHS> {
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

impl<F: Field, const EIGHTHS: usize> SparseDensePrefix<F> for OffsetScalePrefix<EIGHTHS> {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // The offset bits of y (interleaved index bits 0/2/4) stay in the
        // suffix until the final phase; this pairing with the suffix's
        // `b.len() < 6` guard assumes phase boundaries never fall inside the
        // low six index bits.
        debug_assert!(suffix_len == 0 || suffix_len >= 6);
        if suffix_len != 0 {
            return F::one();
        }
        let (_, yb) = b.uninterleave();
        let offset = u64::from(yb) & Self::OFFSET_MASK;
        checkpoints[Self::VARIANT] * F::from_u64(1u64 << ((XLEN / 8) as u64 * offset))
    }
}
