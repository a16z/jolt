use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::jolt::lookup_table::prefixes::Prefixes;

/// (address, offset)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct HalfwordAlignmentTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for HalfwordAlignmentTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        (index % 2 == 0).into()
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

    use crate::jolt::lookup_table::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::HalfwordAlignmentTable;

    #[test]
    fn assert_halfword_alignment_materialize_entry() {
        materialize_entry_test::<Fr, HalfwordAlignmentTable<32>>();
    }

    #[test]
    fn assert_halford_alignment_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, HalfwordAlignmentTable<8>>();
    }

    #[test]
    fn assert_halford_alignment_mle_random() {
        instruction_mle_random_test::<Fr, HalfwordAlignmentTable<32>>();
    }

    #[test]
    fn assert_halfword_alignment_prefix_suffix() {
        prefix_suffix_test::<Fr, HalfwordAlignmentTable<32>>();
    }
}
