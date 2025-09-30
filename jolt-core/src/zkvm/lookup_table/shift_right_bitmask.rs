use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShiftRightBitmaskTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for ShiftRightBitmaskTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let shift = (index % XLEN as u128) as usize;
        let ones = ((1u128 << (XLEN - shift)) - 1) as u64;
        ones << shift
    }

    fn evaluate_mle_field<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let log_w = XLEN.log_2();
        let r = &r[r.len() - log_w..];

        let mut dp = vec![F::zero(); 1 << log_w];

        for s in 0..XLEN {
            let bitmask = ((1u128 << (XLEN - s)) - 1) << s;
            let mut eq_val = F::one();

            for i in 0..log_w {
                let bit = (s >> i) & 1;
                eq_val *= if bit == 0 {
                    F::one() - r[log_w - i - 1]
                } else {
                    r[log_w - i - 1]
                };
            }

            dp[s] = F::from_u128(bitmask) * eq_val;
        }

        dp.into_iter().sum()
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F::Challenge]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let log_w = XLEN.log_2();
        let r = &r[r.len() - log_w..];

        let mut dp = vec![F::zero(); 1 << log_w];

        for s in 0..XLEN {
            let bitmask = ((1u128 << (XLEN - s)) - 1) << s;
            let mut eq_val = F::one();

            for i in 0..log_w {
                let bit = (s >> i) & 1;
                eq_val *= if bit == 0 {
                    F::one() - r[log_w - i - 1]
                } else {
                    r[log_w - i - 1].into()
                };
            }

            dp[s] = F::from_u128(bitmask) * eq_val;
        }

        dp.into_iter().sum()
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ShiftRightBitmaskTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Pow2]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, pow2] = suffixes.try_into().unwrap();
        // 2^XLEN - 2^shift = 0b11...100..0
        F::from_u128(1 << XLEN) * one - prefixes[Prefixes::Pow2] * pow2
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::ShiftRightBitmaskTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ShiftRightBitmaskTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ShiftRightBitmaskTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ShiftRightBitmaskTable<XLEN>>();
    }
}
