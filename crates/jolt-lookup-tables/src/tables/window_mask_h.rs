use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Halfword window mask over a doubleword:
/// `(2^(XLEN/4) − 1) << ((XLEN/8)·(ea mod 8 & !1))`.
///
/// At XLEN = 64 this is `0xFFFF << (8·(ea & 6))`, the byte mask of the
/// halfword at offset `ea mod 8` within its containing doubleword. Only bits
/// 1 and 2 of the (non-interleaved) effective address are read; bit 0 is zero
/// on the halfword-aligned addresses the surrounding sequence asserts, and
/// ignoring it keeps the maximum output `0xFFFF << 48` in `u64` range with a
/// rank-1 decomposition.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WindowMaskHTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for WindowMaskHTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let eighth = XLEN / 8;
        let mask = ((1u128 << (2 * eighth)) - 1) as u64;
        let offset = (index & 6) as u32;
        mask << (eighth as u32 * offset)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let eighth = XLEN / 8;
        let mask = F::from_u128((1u128 << (2 * eighth)) - 1);
        let mut result = mask;
        // offset = 4·b2 + 2·b1 (bit 0 ignored)
        for i in 1..3 {
            let b_i: F = r[r.len() - 1 - i].into();
            let scale = F::from_u128((1u128 << (eighth << i)) - 1);
            result = result + result * (scale * b_i);
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for WindowMaskHTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::Pow2OffsetH]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        // The Pow2Offset prefix/suffix pair hardcodes the 8-bit lane
        // granularity.
        const { assert!(XLEN == 64, "Pow2Offset hardcodes 8-bit lanes") };
        &[Suffixes::Pow2OffsetH]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        const { assert!(XLEN == 64, "Pow2Offset hardcodes 8-bit lanes") };
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2_offset_h] = suffixes.try_into().unwrap();
        F::from_u128((1u128 << (XLEN / 4)) - 1) * prefixes[Prefixes::Pow2OffsetH] * pow2_offset_h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{
        mle_full_hypercube_test, mle_random_test, prefix_suffix_materialization_test,
        prefix_suffix_test,
    };
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, WindowMaskHTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, WindowMaskHTable<XLEN>>();
    }

    /// Two-round phases put a phase boundary inside the low three index bits
    /// (suffix_len hits 2), exercising every placement of bits 2-1 relative
    /// to the phase window in the Pow2OffsetH prefix/suffix pair.
    #[test]
    fn prefix_suffix_small_phases() {
        prefix_suffix_materialization_test::<XLEN, Fr, WindowMaskHTable<XLEN>>(2, 3);
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, WindowMaskHTable<8>>();
    }
}
