use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `P_b = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` over the bits the prefix
/// owns: the offset scale of the second operand. The offset bits y_0..y_2 sit
/// at even index positions 0/2/4.
///
/// Phase-boundary-agnostic: the scale is a product of per-bit factors
/// `1 + (2^((XLEN/8)·2^i) − 1)·y_i`, so each side supplies exactly the bits
/// it owns (the suffix carries partial factors for offset bits below the
/// boundary).
///
/// Raw-index counterpart: [`super::pow2_offset::Pow2OffsetPrefix`] (same
/// scale values over a single-operand index).
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

impl<F: JoltField, const EIGHTHS: usize> SparseDensePrefix<F> for OffsetScalePrefix<EIGHTHS> {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len > 4 {
            // All offset bits are in the suffix, which supplies the factor.
            return F::one();
        }
        let bits: u128 = b.into();
        let mut value = checkpoints[Self::VARIANT];
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
