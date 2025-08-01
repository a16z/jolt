use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::zkvm::lookup_table::prefixes::Prefixes;

/// (address, offset)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct WordAlignmentTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for WordAlignmentTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        index.is_multiple_of(4).into()
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        // The two least significant bits should be 0 for word alignment
        let lsb0 = r[r.len() - 1];
        let lsb1 = r[r.len() - 2];
        (F::one() - lsb0) * (F::one() - lsb1)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for WordAlignmentTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::TwoLsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [two_lsb] = suffixes.try_into().unwrap();
        // Returns 1 if two_lsb == 0 (i.e., both bits are 0), otherwise 0
        prefixes[Prefixes::TwoLsb] * two_lsb
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::instruction_lookups::WORD_SIZE;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    use super::WordAlignmentTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, WordAlignmentTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, WordAlignmentTable<WORD_SIZE>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<WORD_SIZE, Fr, WordAlignmentTable<WORD_SIZE>>();
    }
}
