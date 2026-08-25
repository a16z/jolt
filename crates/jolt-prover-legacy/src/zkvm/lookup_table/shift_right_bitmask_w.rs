use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltLookupTable, PrefixSuffixDecomposition};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShiftRightBitmaskWTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for ShiftRightBitmaskWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let half = XLEN / 2;
        let shift = (index % half as u128) as usize;
        ((1u128 << half) - (1u128 << shift)) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let half = XLEN / 2;
        let log_half = half.trailing_zeros() as usize;
        let mut pow2w = F::one();
        for i in 0..log_half {
            pow2w *= F::one() + F::from_u64((1 << (1 << i)) - 1) * r[r.len() - i - 1];
        }
        F::from_u128(1u128 << half) - pow2w
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ShiftRightBitmaskWTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Pow2W]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, pow2w] = suffixes.try_into().unwrap();
        F::from_u128(1u128 << (XLEN / 2)) * one - prefixes[Prefixes::Pow2W] * pow2w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use ark_bn254::Fr;
    use common::constants::XLEN;

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ShiftRightBitmaskWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ShiftRightBitmaskWTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ShiftRightBitmaskWTable<8>>();
    }

    #[test]
    fn materialize_matches_word_bitmask() {
        for shift in 0..32 {
            assert_eq!(
                ShiftRightBitmaskWTable::<64>.materialize_entry(shift),
                (1u64 << 32) - (1u64 << shift),
            );
        }
    }
}
