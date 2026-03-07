use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::unsigned_less_than::UnsignedLessThanTable;
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnsignedGreaterThanEqualTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for UnsignedGreaterThanEqualTable<XLEN> {
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

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        F::one() - UnsignedLessThanTable::<XLEN>.evaluate_mle::<F, C>(r)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for UnsignedGreaterThanEqualTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        // 1 - LTU(x, y)
        one - prefixes[Prefixes::LessThan] * one - prefixes[Prefixes::Eq] * less_than
    }
}
