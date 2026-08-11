use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `P_b = 2^(8·(y mod 8 & (8−EIGHTHS)))` over bound bits: 1 until the final
/// phase (the interleaved offset bits of `y` sit in the last 5 index bits),
/// then the offset scale of the second operand.
pub enum OffsetScalePrefix<const EIGHTHS: usize> {}

impl<const EIGHTHS: usize> OffsetScalePrefix<EIGHTHS> {
    const OFFSET_MASK: u64 = (8 - EIGHTHS) as u64;

    fn variant() -> Prefixes {
        match EIGHTHS {
            1 => Prefixes::OffsetScaleB,
            2 => Prefixes::OffsetScaleH,
            4 => Prefixes::OffsetScaleW,
            _ => unreachable!(),
        }
    }
}

impl<F: Field, const EIGHTHS: usize> SparseDensePrefix<F> for OffsetScalePrefix<EIGHTHS> {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len != 0 {
            return F::one();
        }
        let (_, yb) = b.uninterleave();
        let offset = u64::from(yb) & Self::OFFSET_MASK;
        checkpoints[Self::variant()] * F::from_u64(1u64 << ((XLEN / 8) as u64 * offset))
    }
}
