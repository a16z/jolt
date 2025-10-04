use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MulUNoOverflowTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for MulUNoOverflowTable<XLEN> {
    /// Returns 1 if unsigned multiplication fits in XLEN bits, 0 if overflow.
    /// Overflow occurs when the upper XLEN bits of the 2*XLEN bit product are non-zero.
    fn materialize_entry(&self, index: u128) -> u64 {
        let upper_bits = index >> XLEN;
        (upper_bits == 0) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut result = F::one();
        for i in 0..XLEN {
            result *= F::one() - r[i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for MulUNoOverflowTable<XLEN> {
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

    use super::MulUNoOverflowTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, MulUNoOverflowTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, MulUNoOverflowTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, MulUNoOverflowTable<XLEN>>();
    }
}
