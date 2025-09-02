use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};
use crate::field::MontU128;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnsignedLessThanEqualTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for UnsignedLessThanEqualTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x <= y).into()
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut lt = F::zero();
        let mut eq = F::one();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            lt += (F::one() - F::from_u128_mont(x_i)).mul_u128_mont_form(y_i) * eq;
            eq *= F::from_u128_mont(x_i) * F::from_u128_mont(y_i) + (F::one() - F::from_u128_mont(x_i)) * (F::one() - F::from_u128_mont(y_i));
        }

        lt + eq
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for UnsignedLessThanEqualTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan, Suffixes::Eq]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than, eq] = suffixes.try_into().unwrap();
        // LT(x, y) + EQ(x, y)
        prefixes[Prefixes::LessThan] * one
            + prefixes[Prefixes::Eq] * less_than
            + prefixes[Prefixes::Eq] * eq
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    use super::UnsignedLessThanEqualTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, UnsignedLessThanEqualTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, UnsignedLessThanEqualTable<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, UnsignedLessThanEqualTable<32>>();
    }
}
