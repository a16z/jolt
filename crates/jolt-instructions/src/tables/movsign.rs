use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Returns all-ones if the MSB of the first operand is set, else zero.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MovsignTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for MovsignTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let sign_bit_pos = 2 * XLEN - 1;
        let sign_bit = 1u128 << sign_bit_pos;
        if index & sign_bit != 0 {
            ((1u128 << XLEN) - 1) as u64
        } else {
            0
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let sign_bit = r[0];
        let ones: u64 = ((1u128 << XLEN) - 1) as u64;
        sign_bit * F::from_u64(ones)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for MovsignTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one] = suffixes.try_into().unwrap();
        let ones: u64 = ((1u128 << XLEN) - 1) as u64;
        F::from_u64(ones) * prefixes[Prefixes::LeftOperandMsb] * one
    }
}
