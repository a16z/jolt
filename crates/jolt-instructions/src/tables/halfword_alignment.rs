use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct HalfwordAlignmentTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for HalfwordAlignmentTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        (index.is_multiple_of(2)).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let lsb = r[r.len() - 1];
        F::one() - lsb
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for HalfwordAlignmentTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Lsb]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lsb] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::Lsb] * lsb
    }
}
