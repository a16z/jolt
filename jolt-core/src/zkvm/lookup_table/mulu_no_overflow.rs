use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MulUNoOverflow<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for MulUNoOverflow<XLEN> {
    /// Returns 1 if unsigned multiplication fits in XLEN bits, 0 if overflow.
    /// Overflow occurs when the upper XLEN bits of the 2*XLEN bit product are non-zero.
    fn materialize_entry(&self, index: u128) -> u64 {
        let upper_bits = index >> XLEN;
        (upper_bits == 0) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut result = F::one();
        for i in 0..XLEN {
            result *= F::one() - r[i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for MulUNoOverflow<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::OverflowBitsZero]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [overflow_bits_zero] = suffixes.try_into().unwrap();
        prefixes[Prefixes::OverflowBitsZero] * overflow_bits_zero
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::MulUNoOverflow;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, MulUNoOverflow<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, MulUNoOverflow<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, MulUNoOverflow<XLEN>>();
    }
}
