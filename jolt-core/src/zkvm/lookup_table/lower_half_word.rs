use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

/// LowerHalfWord table - extracts the lower half of a word
/// For XLEN=64, this extracts the lower 32 bits
/// For XLEN=32, this extracts the lower 16 bits
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct LowerHalfWordTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for LowerHalfWordTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        // Extract the lower half of the word
        let half_word_size = XLEN / 2;
        (index % (1u128 << half_word_size)) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F::Challenge]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let half_word_size = XLEN / 2;
        let mut result = F::zero();
        // Sum the lower half_word_size bits
        for i in 0..half_word_size {
            result += F::from_u64(1 << (half_word_size - 1 - i)) * r[XLEN + half_word_size + i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for LowerHalfWordTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerHalfWord]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_half_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerHalfWord] * one + lower_half_word
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::LowerHalfWordTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, LowerHalfWordTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, LowerHalfWordTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, LowerHalfWordTable<XLEN>>();
    }
}
