use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShiftRightBitmaskTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for ShiftRightBitmaskTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let shift = index % WORD_SIZE as u64;
        let ones = (1 << (WORD_SIZE - shift as usize)) - 1;
        ones << shift
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let log_w = WORD_SIZE.log_2();
        let r = &r[r.len() - log_w..];

        let mut dp = vec![F::zero(); 1 << log_w];

        for s in 0..WORD_SIZE {
            let bitmask = ((1 << (WORD_SIZE - s)) - 1) << s;
            let mut eq_val = F::one();

            for i in 0..log_w {
                let bit = (s >> i) & 1;
                eq_val *= if bit == 0 {
                    F::one() - r[log_w - i - 1]
                } else {
                    r[log_w - i - 1]
                };
            }

            dp[s] = F::from_u64(bitmask) * eq_val;
        }

        dp.into_iter().sum()
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for ShiftRightBitmaskTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Pow2]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, pow2] = suffixes.try_into().unwrap();
        // 2^WORD_SIZE - 2^shift = 0b11...100..0
        F::from_u64(1 << WORD_SIZE) * one - prefixes[Prefixes::Pow2] * pow2
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::ShiftRightBitmaskTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ShiftRightBitmaskTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ShiftRightBitmaskTable<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, ShiftRightBitmaskTable<32>>();
    }
}
