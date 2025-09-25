use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pow2Table<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for Pow2Table<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        1 << (index % XLEN as u128) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F::Challenge]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::one();
        for i in 0..XLEN.log_2() {
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for Pow2Table<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Pow2]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Pow2] * pow2
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::Pow2Table;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, Pow2Table<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, Pow2Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, Pow2Table<XLEN>>();
    }
}
