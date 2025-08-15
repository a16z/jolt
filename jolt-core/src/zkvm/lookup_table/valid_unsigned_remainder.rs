use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidUnsignedRemainderTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for ValidUnsignedRemainderTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (remainder, divisor) = uninterleave_bits(index);
        (divisor == 0 || remainder < divisor).into()
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        let mut divisor_is_zero = F::one();
        let mut lt = F::zero();
        let mut eq = F::one();

        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            divisor_is_zero *= F::one() - y_i;
            lt += (F::one() - x_i) * y_i * eq;
            eq *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }

        lt + divisor_is_zero
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for ValidUnsignedRemainderTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LessThan,
            Suffixes::RightOperandIsZero,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than, right_operand_is_zero] = suffixes.try_into().unwrap();
        // divisor == 0 || remainder < divisor
        prefixes[Prefixes::RightOperandIsZero] * right_operand_is_zero
            + prefixes[Prefixes::LessThan] * one
            + prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::ValidUnsignedRemainderTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ValidUnsignedRemainderTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ValidUnsignedRemainderTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ValidUnsignedRemainderTable<XLEN>>();
    }
}
