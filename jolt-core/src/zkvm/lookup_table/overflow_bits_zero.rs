use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct OverflowBitsZero<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for OverflowBitsZero<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        // Check if upper XLEN bits are all zero
        let upper_bits = index >> XLEN;
        (upper_bits == 0) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);

        // We want to return 1 if all upper bits are 0, otherwise 0
        // This is the product of (1 - r_i) for all upper bits
        let mut result = F::one();
        for i in 0..XLEN {
            result *= F::one() - r[i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for OverflowBitsZero<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::OverflowBitsZero]
    }

    fn combine<F: JoltField>(&self, _prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [overflow_bits_zero] = suffixes.try_into().unwrap();
        // The UpperWordIsZero suffix already computes exactly what we need:
        // 1 if upper word is zero, 0 otherwise
        _prefixes[Prefixes::OverflowBitsZero] * overflow_bits_zero
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::OverflowBitsZero;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, OverflowBitsZero<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, OverflowBitsZero<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, OverflowBitsZero<XLEN>>();
    }
}
