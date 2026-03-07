use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignedLessThanTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for SignedLessThanTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match XLEN {
            #[cfg(test)]
            8 => ((x as i8) < y as i8).into(),
            32 => ((x as i32) < y as i32).into(),
            64 => ((x as i64) < y as i64).into(),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let x_sign = r[0];
        let y_sign = r[1];

        let mut lt = F::zero();
        let mut eq = F::one();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            lt += (F::one() - x_i) * y_i * eq;
            eq *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }

        x_sign - y_sign + lt
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for SignedLessThanTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LeftOperandMsb] * one - prefixes[Prefixes::RightOperandMsb] * one
            + prefixes[Prefixes::LessThan] * one
            + prefixes[Prefixes::Eq] * less_than
    }
}
