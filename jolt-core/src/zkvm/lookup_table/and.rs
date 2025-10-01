use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AndTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for AndTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        x & y
    }
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut result = F::zero();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (XLEN - 1 - i)) * x_i * y_i;
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for AndTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::And]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, and] = suffixes.try_into().unwrap();
        prefixes[Prefixes::And] * one + and
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::AndTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, AndTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, AndTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, AndTable<XLEN>>();
    }
}
