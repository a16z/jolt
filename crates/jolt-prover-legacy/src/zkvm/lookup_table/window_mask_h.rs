use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::zkvm::lookup_table::prefixes::Prefixes;

/// Halfword window mask over a doubleword:
/// `(2^(XLEN/4) − 1) << ((XLEN/8)·(ea mod 8 & !1))`.
///
/// At XLEN = 64 this is `0xFFFF << (8·(ea & 6))`, the byte mask of the
/// halfword at offset `ea mod 8` within its containing doubleword. Only bits
/// 1 and 2 of the (non-interleaved) effective address are read; bit 0 is zero
/// on the halfword-aligned addresses the surrounding sequence asserts, and
/// ignoring it keeps the maximum output `0xFFFF << 48` in `u64` range with a
/// rank-1 decomposition.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct WindowMaskHTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for WindowMaskHTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let eighth = XLEN / 8;
        let mask = ((1u128 << (2 * eighth)) - 1) as u64;
        let offset = (index & 6) as u32;
        mask << (eighth as u32 * offset)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
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
    fn suffixes(&self) -> Vec<Suffixes> {
        // The Pow2Offset prefix/suffix pair hardcodes the 8-bit lane
        // granularity.
        const { assert!(XLEN == 64, "Pow2Offset hardcodes 8-bit lanes") };
        vec![Suffixes::Pow2OffsetH]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        const { assert!(XLEN == 64, "Pow2Offset hardcodes 8-bit lanes") };
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2_offset_h] = suffixes.try_into().unwrap();
        F::from_u128((1u128 << (XLEN / 4)) - 1) * prefixes[Prefixes::Pow2OffsetH] * pow2_offset_h
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::WindowMaskHTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        prefix_suffix_test_with_phase_size,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, WindowMaskHTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, WindowMaskHTable<XLEN>>();
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
        prefix_suffix_test_with_phase_size::<XLEN, Fr, WindowMaskHTable<XLEN>>(2, 20);
    }
}
