use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShiftRightBitmaskTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for ShiftRightBitmaskTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let shift = (index % XLEN as u128) as usize;
        let ones = ((1u128 << (XLEN - shift)) - 1) as u64;
        ones << shift
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let log_w = XLEN.trailing_zeros() as usize;
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

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, pow2] = suffixes.try_into().unwrap();
        F::from_u128(1 << XLEN) * one - prefixes[Prefixes::Pow2] * pow2
    }
}
