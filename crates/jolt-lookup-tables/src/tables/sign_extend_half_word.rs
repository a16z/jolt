use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::XLEN;

/// Sign-extends the lower half of a word to the full word width.
/// For XLEN=64, sign-extends a 32-bit value to 64 bits.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignExtendHalfWordTable;

impl LookupTable for SignExtendHalfWordTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let half_word_size = XLEN / 2;
        let lower_half = (index % (1u128 << half_word_size)) as u64;
        let sign_bit = (lower_half >> (half_word_size - 1)) & 1;

        if sign_bit == 1 {
            lower_half | (((1u64 << half_word_size) - 1) << half_word_size)
        } else {
            lower_half
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let half_word_size = XLEN / 2;

        let mut lower_half = F::zero();
        for i in 0..half_word_size {
            lower_half += F::from_u64(1 << (half_word_size - 1 - i)) * r[XLEN + half_word_size + i];
        }

        let sign_bit = r[XLEN + half_word_size];

        let mut upper_half = F::zero();
        for i in 0..half_word_size {
            upper_half += F::from_u64(1 << (half_word_size - 1 - i)) * sign_bit;
        }

        lower_half + upper_half * F::from_u64(1 << half_word_size)
    }
}

impl PrefixSuffixDecomposition for SignExtendHalfWordTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::LowerHalfWord,
            Suffixes::SignExtensionUpperHalf,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, lower_half_word, sign_extension_upper_half] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerHalfWord] * one
            + lower_half_word
            + prefixes[Prefixes::SignExtensionUpperHalf] * sign_extension_upper_half
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, SignExtendHalfWordTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, SignExtendHalfWordTable>();
    }
}
