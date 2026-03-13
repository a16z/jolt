use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MulUNoOverflowTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for MulUNoOverflowTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let upper_bits = index >> XLEN;
        (upper_bits == 0) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::one();
        for r_i in &r[..XLEN] {
            result *= F::one() - *r_i;
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for MulUNoOverflowTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::OverflowBitsZero]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [overflow_bits_zero] = suffixes.try_into().unwrap();
        prefixes[Prefixes::OverflowBitsZero] * overflow_bits_zero
    }
}
