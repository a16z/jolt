use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (remainder, divisor)
pub struct ValidSignedRemainderTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for ValidSignedRemainderTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match XLEN {
            8 => {
                let (remainder, divisor) = (x as u8 as i8, y as u8 as i8);
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> (XLEN - 1);
                    let divisor_sign = divisor >> (XLEN - 1);
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            32 => {
                let (remainder, divisor) = (x as i32, y as i32);
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> (XLEN - 1);
                    let divisor_sign = divisor >> (XLEN - 1);
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            64 => {
                let (remainder, divisor) = (x as i64, y as i64);
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> (XLEN - 1);
                    let divisor_sign = divisor >> (XLEN - 1);
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
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

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ValidSignedRemainderTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LessThan,
            Suffixes::GreaterThan,
            Suffixes::LeftOperandIsZero,
            Suffixes::RightOperandIsZero,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than, greater_than, left_operand_is_zero, right_operand_is_zero] =
            suffixes.try_into().unwrap();
        // divisor == 0 || (isPositive(remainder) && remainder <= diviisor) || (isNegative(divisor) && divisor >= remainder)
        prefixes[Prefixes::RightOperandIsZero] * right_operand_is_zero
            + prefixes[Prefixes::PositiveRemainderEqualsDivisor] * less_than
            + prefixes[Prefixes::PositiveRemainderLessThanDivisor] * one
            + prefixes[Prefixes::NegativeDivisorZeroRemainder] * left_operand_is_zero
            + prefixes[Prefixes::NegativeDivisorEqualsRemainder] * greater_than
            + prefixes[Prefixes::NegativeDivisorGreaterThanRemainder] * one
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::ValidSignedRemainderTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ValidSignedRemainderTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ValidSignedRemainderTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ValidSignedRemainderTable<XLEN>>();
    }
}
