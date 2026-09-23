use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// `2^(8·offset)` where `offset` is the low 3 bits of the (non-interleaved)
/// lookup index with bits below `LOW_BIT` zeroed: the lane-offset scale of a
/// doubleword effective address. `LOW_BIT = 2` is the word variant (bit 2
/// only), `1` the halfword variant (bits 2-1), `0` the byte variant (bits
/// 2-0).
pub enum Pow2OffsetPrefix<const LOW_BIT: usize> {}

impl<const LOW_BIT: usize> Pow2OffsetPrefix<LOW_BIT> {
    /// Which of the low 3 index bits contribute to the offset.
    const OFFSET_MASK: u128 = ((7 >> LOW_BIT) << LOW_BIT) as u128;

    const VARIANT: Prefixes = match LOW_BIT {
        0 => Prefixes::Pow2OffsetB,
        1 => Prefixes::Pow2OffsetH,
        2 => Prefixes::Pow2OffsetW,
        _ => panic!("unsupported LOW_BIT"),
    };
}

impl<const LOW_BIT: usize, F: JoltField> SparseDensePrefix<F> for Pow2OffsetPrefix<LOW_BIT> {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // Phase-boundary-agnostic split of the index around the low 3 offset
        // bits: the suffix owns bits [0, suffix_len), this phase's bits `b`
        // own [suffix_len, suffix_len + b.len()), and everything above is
        // bound into the checkpoint. The offset factors multiply per bit, so
        // each side supplies exactly the bits it owns (the suffixes carry
        // partial factors for offset bits below the boundary).
        if suffix_len >= 3 {
            // All offset bits are in the suffix, which supplies the factor.
            return F::one();
        }
        let bits: u128 = b.into();
        let offset = ((bits << suffix_len) & Self::OFFSET_MASK) as u32;
        checkpoints[Self::VARIANT] * F::from_u64(1u64 << (8 * offset))
    }
}
