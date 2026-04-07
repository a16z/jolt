use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;
use crate::XLEN;

/// (remainder, divisor)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidSignedRemainderTable;

impl LookupTable for ValidSignedRemainderTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let (remainder, divisor) = (x as i64, y as i64);
        if remainder == 0 || divisor == 0 {
            1
        } else {
            let remainder_sign = remainder >> (XLEN - 1);
            let divisor_sign = divisor >> (XLEN - 1);
            (remainder.unsigned_abs() < divisor.unsigned_abs() && remainder_sign == divisor_sign)
                .into()
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let x_sign = r[0];
        let y_sign = r[1];

        let mut remainder_is_zero = F::one() - r[0];
        let mut divisor_is_zero = F::one() - r[1];
        let mut positive_remainder_equals_divisor = (F::one() - x_sign) * (F::one() - y_sign);
        let mut positive_remainder_less_than_divisor = (F::one() - x_sign) * (F::one() - y_sign);
        let mut negative_divisor_equals_remainder = x_sign * y_sign;
        let mut negative_divisor_greater_than_remainder = x_sign * y_sign;

        for i in 1..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            if i == 1 {
                positive_remainder_less_than_divisor *= (F::one() - x_i) * y_i;
                negative_divisor_greater_than_remainder *= x_i * (F::one() - y_i);
            } else {
                positive_remainder_less_than_divisor +=
                    positive_remainder_equals_divisor * (F::one() - x_i) * y_i;
                negative_divisor_greater_than_remainder +=
                    negative_divisor_equals_remainder * x_i * (F::one() - y_i);
            }
            positive_remainder_equals_divisor *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
            negative_divisor_equals_remainder *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
            remainder_is_zero *= F::one() - x_i;
            divisor_is_zero *= F::one() - y_i;
        }

        positive_remainder_less_than_divisor
            + negative_divisor_greater_than_remainder
            + y_sign * remainder_is_zero
            + divisor_is_zero
    }
}

impl PrefixSuffixDecomposition for ValidSignedRemainderTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::LessThan,
            Suffixes::GreaterThan,
            Suffixes::LeftOperandIsZero,
            Suffixes::RightOperandIsZero,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than, greater_than, left_operand_is_zero, right_operand_is_zero] =
            suffixes.try_into().unwrap();
        prefixes[Prefixes::RightOperandIsZero] * right_operand_is_zero
            + prefixes[Prefixes::PositiveRemainderEqualsDivisor] * less_than
            + prefixes[Prefixes::PositiveRemainderLessThanDivisor] * one
            + prefixes[Prefixes::NegativeDivisorZeroRemainder] * left_operand_is_zero
            + prefixes[Prefixes::NegativeDivisorEqualsRemainder] * greater_than
            + prefixes[Prefixes::NegativeDivisorGreaterThanRemainder] * one
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, ValidSignedRemainderTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, ValidSignedRemainderTable>();
    }
}
