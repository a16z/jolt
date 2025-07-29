use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (divisor, quotient)
pub struct ValidDiv0Table<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for ValidDiv0Table<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (divisor, quotient) = uninterleave_bits(index);
        if divisor == 0 {
            match WORD_SIZE {
                8 => (quotient == u8::MAX as u64).into(),
                32 => (quotient == u32::MAX as u64).into(),
                _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
            }
        } else {
            1
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        let mut divisor_is_zero = F::one();
        let mut is_valid_div_by_zero = F::one();

        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            divisor_is_zero *= F::one() - x_i;
            is_valid_div_by_zero *= (F::one() - x_i) * y_i;
        }

        F::one() - divisor_is_zero + is_valid_div_by_zero
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for ValidDiv0Table<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LeftOperandIsZero,
            Suffixes::DivByZero,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, left_operand_is_zero, div_by_zero] = suffixes.try_into().unwrap();
        // If the divisor is *not* zero, both:
        // `prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero` and
        // `prefixes[Prefixes::DivByZero] * div_by_zero`
        // will be zero.
        //
        // If the divisor *is* zero, returns 1 (on the Boolean hypercube)
        // iff the quotient is valid (i.e. 2^WORD_SIZE - 1).
        one - prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero
            + prefixes[Prefixes::DivByZero] * div_by_zero
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    use super::ValidDiv0Table;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ValidDiv0Table<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ValidDiv0Table<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, ValidDiv0Table<32>>();
    }
}
