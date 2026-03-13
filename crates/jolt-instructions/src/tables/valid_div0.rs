use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// (divisor, quotient)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidDiv0Table<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for ValidDiv0Table<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (divisor, quotient) = uninterleave_bits(index);
        if divisor == 0 {
            match XLEN {
                #[cfg(test)]
                8 => (quotient == u8::MAX as u64).into(),
                32 => (quotient == u32::MAX as u64).into(),
                64 => (quotient == u64::MAX).into(),
                _ => panic!("{XLEN}-bit word size is unsupported"),
            }
        } else {
            1
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let mut divisor_is_zero = F::one();
        let mut is_valid_div_by_zero = F::one();

        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            divisor_is_zero *= F::one() - x_i;
            is_valid_div_by_zero *= (F::one() - x_i) * y_i;
        }

        F::one() - divisor_is_zero + is_valid_div_by_zero
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ValidDiv0Table<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LeftOperandIsZero,
            Suffixes::DivByZero,
        ]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, left_operand_is_zero, div_by_zero] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero
            + prefixes[Prefixes::DivByZero] * div_by_zero
    }
}
