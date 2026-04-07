use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;
use crate::XLEN;

/// (divisor, quotient)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidDiv0Table;

impl LookupTable for ValidDiv0Table {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (divisor, quotient) = uninterleave_bits(index);
        if divisor == 0 {
            (quotient == u64::MAX).into()
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

impl PrefixSuffixDecomposition for ValidDiv0Table {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::LeftOperandIsZero,
            Suffixes::DivByZero,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, left_operand_is_zero, div_by_zero] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero
            + prefixes[Prefixes::DivByZero] * div_by_zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, ValidDiv0Table>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, ValidDiv0Table>();
    }
}
