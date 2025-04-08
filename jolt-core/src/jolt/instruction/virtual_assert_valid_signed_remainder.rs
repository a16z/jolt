use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (remainder, divisor)
pub struct AssertValidSignedRemainderInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for AssertValidSignedRemainderInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn lookup_entry(&self) -> u64 {
        match WORD_SIZE {
            32 => {
                let remainder = self.0 as u32 as i32;
                let divisor = self.1 as u32 as i32;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 31;
                    let divisor_sign = divisor >> 31;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            64 => {
                let remainder = self.0 as i64;
                let divisor = self.1 as i64;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 63;
                    let divisor_sign = divisor >> 63;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
        }
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match WORD_SIZE {
            8 => {
                let (remainder, divisor) = (x as u8 as i8, y as u8 as i8);
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> (WORD_SIZE - 1);
                    let divisor_sign = divisor >> (WORD_SIZE - 1);
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
                    let remainder_sign = remainder >> (WORD_SIZE - 1);
                    let divisor_sign = divisor >> (WORD_SIZE - 1);
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8), rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64, rng.next_u32() as u64),
            64 => Self(rng.next_u64(), rng.next_u64()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
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

        for i in 1..WORD_SIZE {
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

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for AssertValidSignedRemainderInstruction<WORD_SIZE>
{
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

    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::AssertValidSignedRemainderInstruction;

    #[test]
    fn assert_valid_signed_remainder_materialize_entry() {
        materialize_entry_test::<Fr, AssertValidSignedRemainderInstruction<32>>();
    }

    #[test]
    fn assert_valid_signed_remainder_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, AssertValidSignedRemainderInstruction<8>>();
    }

    #[test]
    fn assert_valid_signed_remainder_mle_random() {
        instruction_mle_random_test::<Fr, AssertValidSignedRemainderInstruction<32>>();
    }

    #[test]
    fn assert_valid_signed_remainder_prefix_suffix() {
        prefix_suffix_test::<Fr, AssertValidSignedRemainderInstruction<32>>();
    }
}
