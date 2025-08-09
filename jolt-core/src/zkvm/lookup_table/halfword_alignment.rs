use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::zkvm::lookup_table::prefixes::Prefixes;

/// (address, offset)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct HalfwordAlignmentTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for HalfwordAlignmentTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        index.is_multiple_of(2).into()
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        let lsb = r[r.len() - 1];
        F::one() - lsb
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for HalfwordAlignmentTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Lsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lsb] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::Lsb] * lsb
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::HalfwordAlignmentTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, HalfwordAlignmentTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, HalfwordAlignmentTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, HalfwordAlignmentTable<XLEN>>();
    }
}
