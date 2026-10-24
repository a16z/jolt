use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Doubleword window mask truncated to the containing doubleword:
/// `(2^XLEN − 1) << ((XLEN/8)·(ea mod 8)) mod 2^XLEN = 2^XLEN − 2^((XLEN/8)·(ea mod 8))`.
///
/// At XLEN = 64 this sets the bytes at offsets `ea mod 8` through 7. A
/// misaligned access splits its bytes around this mask: the containing
/// doubleword's bytes inside it, and the next doubleword's bytes below it.
/// Reads only the low 3 bits of the (non-interleaved) effective address.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WindowMaskDTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for WindowMaskDTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let all_ones = (1u128 << XLEN) - 1;
        let offset = (index & 7) as u32;
        ((all_ones << ((XLEN / 8) as u32 * offset)) & all_ones) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let eighth = XLEN / 8;
        // 2^XLEN − 2^(eighth·offset) with offset = 4·b2 + 2·b1 + b0, the power
        // as the product of per-bit factors (1 + (2^(eighth·2^i) − 1)·b_i).
        let mut scale = F::one();
        for i in 0..3 {
            let b_i: F = r[r.len() - 1 - i].into();
            scale = scale + scale * (F::from_u128((1u128 << (eighth << i)) - 1) * b_i);
        }
        F::from_u128(1u128 << XLEN) - scale
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for WindowMaskDTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::Pow2OffsetB]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        // The Pow2Offset prefix/suffix pair hardcodes the 8-bit lane
        // granularity.
        const { assert!(XLEN == 64, "Pow2Offset hardcodes 8-bit lanes") };
        &[Suffixes::One, Suffixes::Pow2OffsetB]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        const { assert!(XLEN == 64, "Pow2Offset hardcodes 8-bit lanes") };
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, pow2_offset_b] = suffixes.try_into().unwrap();
        F::from_u128(1u128 << XLEN) * one - prefixes[Prefixes::Pow2OffsetB] * pow2_offset_b
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
        mle_random_test::<XLEN, Fr, WindowMaskDTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, WindowMaskDTable<XLEN>>();
    }

    /// Two-round phases put a phase boundary inside the low three index bits,
    /// exercising every placement of bits 2-0 relative to the phase window.
    #[test]
    fn prefix_suffix_small_phases() {
        prefix_suffix_materialization_test::<XLEN, Fr, WindowMaskDTable<XLEN>>(2, 3);
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, WindowMaskDTable<8>>();
    }
}
