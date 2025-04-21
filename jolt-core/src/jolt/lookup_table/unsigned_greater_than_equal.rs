use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::SuffixEval,
    unsigned_less_than::UnsignedLessThanTable,
    JoltLookupTable, PrefixSuffixDecomposition,
};
use crate::{field::JoltField, jolt::lookup_table::suffixes::Suffixes, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnsignedGreaterThanEqualTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for UnsignedGreaterThanEqualTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x >= y).into(),
            32 => (x >= y).into(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        F::one() - UnsignedLessThanTable::<WORD_SIZE>::default().evaluate_mle::<F>(r)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for UnsignedGreaterThanEqualTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        // 1 - LTU(x, y)
        one - prefixes[Prefixes::LessThan] * one - prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::jolt::lookup_table::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::UnsignedGreaterThanEqualTable;

    #[test]
    fn bgeu_materialize_entry() {
        materialize_entry_test::<Fr, UnsignedGreaterThanEqualTable<32>>();
    }

    #[test]
    fn bgeu_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, UnsignedGreaterThanEqualTable<8>>();
    }

    #[test]
    fn bgeu_mle_random() {
        instruction_mle_random_test::<Fr, UnsignedGreaterThanEqualTable<32>>();
    }

    #[test]
    fn bgeu_prefix_suffix() {
        prefix_suffix_test::<Fr, UnsignedGreaterThanEqualTable<32>>();
    }
}
