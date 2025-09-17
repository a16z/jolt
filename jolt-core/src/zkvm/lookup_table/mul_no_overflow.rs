use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MulNoOverflowTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for MulNoOverflowTable<XLEN> {
    /// Returns 1 if signed multiplication has no overflow, 0 otherwise.
    /// No overflow when all XLEN upper bits match the sign bit (bit XLEN-1).
    fn materialize_entry(&self, index: u128) -> u64 {
        let sign_bit = (index >> (XLEN - 1)) & 1;
        let upper_bits = index >> XLEN;

        if sign_bit == 0 {
            (upper_bits == 0) as u64
        } else {
            let mask = (1u128 << XLEN) - 1;
            (upper_bits == mask) as u64
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut positive_case = F::one() - r[XLEN];
        for i in 0..XLEN {
            positive_case *= F::one() - r[i];
        }
        let mut negative_case = r[XLEN];
        for i in 0..XLEN {
            negative_case *= r[i];
        }
        positive_case + negative_case
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for MulNoOverflowTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            // When sign bit is 0 and overflow bits are 0
            Suffixes::SignedOverflowBitsZero,
            // When sign bit is 1 and overflow bits are 1
            Suffixes::SignedOverflowBitsOne,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [signed_overflow_bits_zero, signed_overflow_bits_one] = suffixes.try_into().unwrap();
        let positive_case = prefixes[Prefixes::SignedOverflowBitsZero] * signed_overflow_bits_zero;
        let negative_case = prefixes[Prefixes::SignedOverflowBitsOne] * signed_overflow_bits_one;
        positive_case + negative_case
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::MulNoOverflowTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, MulNoOverflowTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, MulNoOverflowTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, MulNoOverflowTable<XLEN>>();
    }
}
