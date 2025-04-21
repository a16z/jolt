use serde::{Deserialize, Serialize};

use super::equal::EqualTable;
use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct NotEqualTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for NotEqualTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x != y).into(),
            32 => (x != y).into(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        F::one() - EqualTable::<WORD_SIZE>::default().evaluate_mle::<F>(r)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for NotEqualTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Eq]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, eq] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::Eq] * eq
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::jolt::lookup_table::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::NotEqualTable;

    #[test]
    fn bne_materialize_entry() {
        materialize_entry_test::<Fr, NotEqualTable<32>>();
    }

    #[test]
    fn bne_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, NotEqualTable<8>>();
    }

    #[test]
    fn bne_mle_random() {
        instruction_mle_random_test::<Fr, NotEqualTable<32>>();
    }

    #[test]
    fn bne_prefix_suffix() {
        prefix_suffix_test::<Fr, NotEqualTable<32>>();
    }
}
