use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::signed_less_than::SignedLessThanTable;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignedGreaterThanEqualTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for SignedGreaterThanEqualTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match XLEN {
            #[cfg(test)]
            8 => (x as i8 >= y as i8).into(),
            32 => (x as i32 >= y as i32).into(),
            64 => (x as i64 >= y as i64).into(),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        F::one() - SignedLessThanTable::<XLEN>.evaluate_mle(r)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for SignedGreaterThanEqualTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        // 1 - LT(x, y) = 1 - (isNegative(x) && isPositive(y)) - LTU(x, y)
        one + prefixes[Prefixes::RightOperandMsb] * one
            - prefixes[Prefixes::LeftOperandMsb] * one
            - prefixes[Prefixes::LessThan] * one
            - prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::SignedGreaterThanEqualTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, SignedGreaterThanEqualTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, SignedGreaterThanEqualTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, SignedGreaterThanEqualTable<XLEN>>();
    }
}
