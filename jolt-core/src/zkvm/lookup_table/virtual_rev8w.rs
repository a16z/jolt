use std::array;
use std::iter;

use serde::{Deserialize, Serialize};
use tracer::instruction::virtual_rev8w::rev8w;

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{IntoField, JoltField};
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualRev8WTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for VirtualRev8WTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        rev8w(index as u64)
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F::Challenge]) -> F {
        let mut bits = r.iter().rev();
        let mut bytes = iter::from_fn(|| {
            let bit_chunk = (&mut bits).take(8).enumerate();
            Some(bit_chunk.map(|(i, b)| b.into_F().mul_u64(1 << i)).sum::<F>())
        });

        // Reverse the bytes in each 32-bit word. i.e.
        //   abcd:efgh => dcba:hgfe
        let [a, b, c, d, e, f, g, h] = array::from_fn(|_| bytes.next().unwrap());
        [d, c, b, a, h, g, f, e]
            .iter()
            .enumerate()
            .map(|(i, b)| b.mul_u64(1 << (i * 8)))
            .sum()
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualRev8WTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Rev8W]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, rev8w] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Rev8W] * one + rev8w
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::VirtualRev8WTable;
    use crate::zkvm::lookup_table::test::lookup_table_mle_random_test;
    use crate::zkvm::lookup_table::test::prefix_suffix_test;
    use common::constants::XLEN;

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualRev8WTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualRev8WTable<XLEN>>();
    }
}
