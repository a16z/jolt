use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::SuffixEval,
    unsigned_less_than::UnsignedLessThanTable,
    JoltLookupTable, PrefixSuffixDecomposition,
};
use crate::{field::JoltField, utils::uninterleave_bits, zkvm::lookup_table::suffixes::Suffixes};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnsignedGreaterThanEqualTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for UnsignedGreaterThanEqualTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match XLEN {
            #[cfg(test)]
            8 => (x >= y).into(),
            32 => (x >= y).into(),
            64 => (x >= y).into(),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        F::one() - UnsignedLessThanTable::<XLEN>.evaluate_mle::<F>(r)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for UnsignedGreaterThanEqualTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        // 1 - LTU(x, y)
        one - prefixes[Prefixes::LessThan] * one - prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::UnsignedGreaterThanEqualTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, UnsignedGreaterThanEqualTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, UnsignedGreaterThanEqualTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, UnsignedGreaterThanEqualTable<XLEN>>();
    }
}
