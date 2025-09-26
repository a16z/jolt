use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::zkvm::lookup_table::prefixes::Prefixes;

/// Pow2W table - computes 2^(x % 32) for VirtualPow2W and VirtualPow2IW instructions
/// Always uses modulo 32 regardless of XLEN
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pow2WTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for Pow2WTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        1 << (index % 32) as u64
    }

    fn evaluate_mle_field<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);
        // We only care about the last 5 bits of the second operand (for modulo 32)
        let mut result = F::one();
        for i in 0..5 {
            // 5 bits for 32 values
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F::Challenge]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);
        // We only care about the last 5 bits of the second operand (for modulo 32)
        let mut result = F::one();
        for i in 0..5 {
            // 5 bits for 32 values
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for Pow2WTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Pow2W]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2w] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Pow2W] * pow2w
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::Pow2WTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, Pow2WTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, Pow2WTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, Pow2WTable<XLEN>>();
    }
}
