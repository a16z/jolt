use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::zkvm::lookup_table::prefixes::Prefixes;

/// Word window mask over a doubleword: `(2^(XLEN/2) − 1) << ((XLEN/2)·ea_2)`
/// where `ea_2` is bit 2 of the (non-interleaved) effective address.
///
/// On a word-aligned address (bits 0–1 zero, enforced separately by
/// `AssertWordAlignment`), this is the byte mask of the word at offset
/// `ea mod 8` within its containing doubleword. Only bit 2 is read, so the
/// output is total and never overflows `u64`.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct WindowMaskWTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for WindowMaskWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let half = XLEN / 2;
        let mask = ((1u128 << half) - 1) as u64;
        let bit2 = ((index >> 2) & 1) as u32;
        mask << (half as u32 * bit2)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let half = XLEN / 2;
        let mask = F::from_u128((1u128 << half) - 1);
        let bit2: F = r[r.len() - 3].into();
        // mask · 2^(half·bit2) = mask · (1 + (2^half − 1)·bit2)
        mask + mask * (F::from_u128((1u128 << half) - 1) * bit2)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for WindowMaskWTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        // The Pow2OffsetW prefix/suffix pair hardcodes the 32-bit lane width,
        // so any other instantiation is a compile error (materialize_entry
        // and evaluate_mle are genuinely XLEN-generic; the decomposition is
        // not).
        const { assert!(XLEN == 64, "Pow2OffsetW hardcodes the 32-bit lane") };
        vec![Suffixes::Pow2OffsetW]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        const { assert!(XLEN == 64, "Pow2OffsetW hardcodes the 32-bit lane") };
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2_offset_w] = suffixes.try_into().unwrap();
        F::from_u128((1u128 << (XLEN / 2)) - 1) * prefixes[Prefixes::Pow2OffsetW] * pow2_offset_w
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::WindowMaskWTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        prefix_suffix_test_with_phase_size,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, WindowMaskWTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, WindowMaskWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, WindowMaskWTable<XLEN>>();
    }

    /// Two-round phases put a phase boundary inside the low three index bits
    /// (suffix_len hits 2), exercising every placement of bit 2 relative to
    /// the phase window in the Pow2OffsetW prefix/suffix pair.
    #[test]
    fn prefix_suffix_small_phases() {
        prefix_suffix_test_with_phase_size::<XLEN, Fr, WindowMaskWTable<XLEN>>(2, 20);
    }
}
