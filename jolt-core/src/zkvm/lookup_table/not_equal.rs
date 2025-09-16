use serde::{Deserialize, Serialize};

use super::equal::EqualTable;
use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::MontU128;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct NotEqualTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for NotEqualTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match XLEN {
            #[cfg(test)]
            8 => (x != y).into(),
            32 => (x != y).into(),
            64 => (x != y).into(),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        F::one() - EqualTable::<XLEN>.evaluate_mle::<F>(r)
    }
    fn evaluate_mle_field<F: JoltField>(&self, r: &[F]) -> F {
        F::one() - EqualTable::<XLEN>.evaluate_mle_field::<F>(r)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for NotEqualTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Eq]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, eq] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::Eq] * eq
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::NotEqualTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, NotEqualTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, NotEqualTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, NotEqualTable<XLEN>>();
    }
}
