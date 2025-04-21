use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecomposition,
};
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct EqualTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for EqualTable<WORD_SIZE> {
    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert!(r.len() % 2 == 0);

        let x = r.iter().step_by(2);
        let y = r.iter().skip(1).step_by(2);
        x.zip(y)
            .map(|(x_i, y_i)| *x_i * y_i + (F::one() - x_i) * (F::one() - y_i))
            .product()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x == y).into()
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for EqualTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Eq]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [eq] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Eq] * eq
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::jolt::lookup_table::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::EqualTable;

    #[test]
    fn beq_prefix_suffix() {
        prefix_suffix_test::<Fr, EqualTable<32>>();
    }

    #[test]
    fn beq_materialize_entry() {
        materialize_entry_test::<Fr, EqualTable<32>>();
    }

    #[test]
    fn beq_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, EqualTable<8>>();
    }

    #[test]
    fn beq_mle_random() {
        instruction_mle_random_test::<Fr, EqualTable<32>>();
    }
}
