use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable,
    PrefixSuffixDecomposition,
};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

/// SignExtendHalfWord table - sign-extends the lower half of a word to full word
/// For XLEN=64, this sign-extends a 32-bit value to 64-bit
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignExtendHalfWordTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for SignExtendHalfWordTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let half_word_size = XLEN / 2;
        let lower_half = (index % (1u128 << half_word_size)) as u64;

        // Check sign bit (bit at position half_word_size - 1)
        let sign_bit = (lower_half >> (half_word_size - 1)) & 1;

        if sign_bit == 1 {
            // Sign extend with 1s
            lower_half | (((1u64 << half_word_size) - 1) << half_word_size)
        } else {
            // Sign extend with 0s (just the lower half)
            lower_half
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let half_word_size = XLEN / 2;

        // Sum for lower half bits (from the second operand, starting at XLEN)
        let mut lower_half = F::zero();
        for i in 0..half_word_size {
            lower_half += F::from_u64(1 << (half_word_size - 1 - i)) * r[XLEN + half_word_size + i];
        }

        let sign_bit = r[XLEN + half_word_size];

        // Upper half: all sign bits
        let mut upper_half = F::zero();
        for i in 0..half_word_size {
            upper_half += F::from_u64(1 << (half_word_size - 1 - i)) * sign_bit;
        }

        lower_half + upper_half * F::from_u64(1 << half_word_size)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for SignExtendHalfWordTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LowerHalfWord,
            Suffixes::SignExtensionUpperHalf,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_half_word, sign_extension_upper_half] = suffixes.try_into().unwrap();

        prefixes[Prefixes::LowerHalfWord] * one
            + lower_half_word
            + prefixes[Prefixes::SignExtensionUpperHalf] * sign_extension_upper_half
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use common::constants::XLEN;

    use super::SignExtendHalfWordTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test,
        lookup_table_mle_random_test,
        prefix_suffix_test,
    };

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, SignExtendHalfWordTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, SignExtendHalfWordTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, SignExtendHalfWordTable<XLEN>>();
    }
}
