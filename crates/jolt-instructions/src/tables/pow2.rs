use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Computes `2^(index % XLEN)`.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pow2Table<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for Pow2Table<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        1 << (index % XLEN as u128) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let log_xlen = XLEN.trailing_zeros() as usize;
        let mut result = F::one();
        for i in 0..log_xlen {
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for Pow2Table<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Pow2]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [pow2] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Pow2] * pow2
    }
}
