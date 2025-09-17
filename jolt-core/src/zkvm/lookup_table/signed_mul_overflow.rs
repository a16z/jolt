use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignedMulOverflow<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for SignedMulOverflow<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        // Check if the upper bits match sign extension
        // Get the sign bit from the lower XLEN bits
        let sign_bit = (index >> (XLEN - 1)) & 1;
        let upper_bits = index >> XLEN;
        
        if sign_bit == 0 {
            // Positive number: upper bits should be all 0s
            (upper_bits == 0) as u64
        } else {
            // Negative number: upper bits should be all 1s
            // For a 2*XLEN bit value, the upper XLEN bits should all be 1
            let mask = (1u128 << XLEN) - 1;
            (upper_bits == mask) as u64
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);
        
        // r[0..XLEN] contains the upper bits
        // r[XLEN..2*XLEN] contains the lower bits
        // The sign bit is the MSB of the lower XLEN bits, which is r[XLEN]
        // let sign_bit = r[XLEN];
        
        // For positive numbers (sign bit = 0): all upper bits should be 0
        // This is: (1 - sign_bit) * product((1 - r[i]) for i in upper bits)
        let mut positive_case = F::one();
        for i in 0..XLEN {
            positive_case *= F::one() - r[i];
        }

        eprintln!("evaluate_mle() before sign_bit: {}", positive_case);
        positive_case *= F::one() - r[XLEN];

        let sign_bit = r[XLEN];
        
        // For negative numbers (sign bit = 1): all upper bits should be 1
        // This is: sign_bit * product(r[i] for i in upper bits)
        let mut negative_case = sign_bit;
        for i in 0..XLEN {
            negative_case *= r[i];
        }
        eprintln!("evaluate_mle(): positive case: {} - negative case: {} - result: {}", positive_case, negative_case, positive_case + negative_case);
        eprintln!("------------------------------------------------------------------------------------------------------------------------------------------------");
        positive_case + negative_case
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for SignedMulOverflow<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        // Two suffixes corresponding to the two terms in the MLE:
        // 1. SignedOverflowBitsZero: for when sign bit is 0 and overflow bits are 0
        // 2. SignedOverflowBitsOne: for when sign bit is 1 and overflow bits are 1
        vec![Suffixes::SignedOverflowBitsZero, Suffixes::SignedOverflowBitsOne]
    }

    fn combine<F: JoltField>(&self, _prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [signed_overflow_bits_zero, signed_overflow_bits_one] = suffixes.try_into().unwrap();
        
        // First term: (1-a_63) * product((1-a_i) for i in 64..127)
        // This is handled by SignedOverflowBitsZero prefix and suffix
        let positive_case = _prefixes[Prefixes::SignedOverflowBitsZero] * signed_overflow_bits_zero;
        
        // Second term: a_63 * product(a_i for i in 64..127)
        // This is handled by SignedOverflowBitsOne prefix and suffix  
        let negative_case = _prefixes[Prefixes::SignedOverflowBitsOne] * signed_overflow_bits_one;

        eprintln!("Specific: {}", _prefixes[Prefixes::SignedOverflowBitsOne]);
        eprintln!("combine(): positive case: {} - negative case: {}", positive_case, negative_case);
        eprintln!("------------------------------------------------------------------------------------------------------------------------------------------------");
        positive_case + negative_case
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::SignedMulOverflow;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, SignedMulOverflow<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, SignedMulOverflow<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, SignedMulOverflow<XLEN>>();
    }
}
