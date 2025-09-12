use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecomposition,
};
use crate::field::MontU128;
use crate::{field::JoltField, utils::uninterleave_bits};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnsignedLessThanTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for UnsignedLessThanTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x < y).into()
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        // \sum_i (1 - x_i) * y_i * \prod_{j < i} ((1 - x_j) * (1 - y_j) + x_j * y_j)
        let mut result = F::zero();
        let mut eq_term = F::one();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += (F::one() - F::from_u128_mont(x_i)).mul_u128_mont_form(y_i) * eq_term;
            eq_term *= F::from_u128_mont(x_i) * F::from_u128_mont(y_i)
                + (F::one() - F::from_u128_mont(x_i)) * (F::one() - F::from_u128_mont(y_i));
        }
        result
    }

    fn evaluate_mle_field<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        // \sum_i (1 - x_i) * y_i * \prod_{j < i} ((1 - x_j) * (1 - y_j) + x_j * y_j)
        let mut result = F::zero();
        let mut eq_term = F::one();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += (F::one() - x_i) * y_i * eq_term;
            eq_term *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for UnsignedLessThanTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LessThan] * one + prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test,
        lookup_table_mle_random_test,
        // prefix_suffix_test,
    };

    use super::UnsignedLessThanTable;

    // #[test]
    // fn prefix_suffix() {
    //     prefix_suffix_test::<Fr, UnsignedLessThanTable<32>>();
    // }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, UnsignedLessThanTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, UnsignedLessThanTable<32>>();
    }
}
