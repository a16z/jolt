use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Word window mask over a doubleword: `(2^(XLEN/2) − 1) << ((XLEN/2)·ea_2)`
/// where `ea_2` is bit 2 of the (non-interleaved) effective address.
///
/// On a word-aligned address (bits 0–1 zero, enforced separately by
/// `AssertWordAlignment`), this is the byte mask of the word at offset
/// `ea mod 8` within its containing doubleword. Only bit 2 is read, so the
/// output is total and never overflows `u64`.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WindowMaskWTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for WindowMaskWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let half = XLEN / 2;
        let mask = ((1u128 << half) - 1) as u64;
        let bit2 = ((index >> 2) & 1) as u32;
        mask << (half as u32 * bit2)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
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
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::Pow2OffsetW]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        // The Pow2OffsetW prefix/suffix pair hardcodes the 32-bit lane width.
        debug_assert_eq!(XLEN, 64);
        &[Suffixes::Pow2OffsetW]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2_offset_w] = suffixes.try_into().unwrap();
        F::from_u128((1u128 << (XLEN / 2)) - 1) * prefixes[Prefixes::Pow2OffsetW] * pow2_offset_w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, WindowMaskWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, WindowMaskWTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, WindowMaskWTable<8>>();
    }
}
