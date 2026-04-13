use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;
use crate::XLEN;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualChangeDivisorTable;

impl LookupTable for VirtualChangeDivisorTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (dividend, divisor) = uninterleave_bits(index);
        let dividend = dividend as i64;
        let divisor = divisor as i64;
        if dividend == i64::MIN && divisor == -1 {
            1
        } else {
            divisor as u64
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut divisor_value = F::zero();
        for i in 0..XLEN {
            let bit_value = r[2 * i + 1];
            let shift = XLEN - 1 - i;
            divisor_value += F::from_u128(1u128 << shift) * bit_value;
        }

        let mut x_product = r[0].into();
        for i in 1..XLEN {
            x_product *= F::one() - r[2 * i];
        }

        let mut y_product = F::one();
        for i in 0..XLEN {
            y_product = y_product * r[2 * i + 1];
        }

        let adjustment = F::from_u64(2) - F::from_u128(1u128 << XLEN);

        divisor_value + x_product * y_product * adjustment
    }
}

impl PrefixSuffixDecomposition for VirtualChangeDivisorTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::RightOperand,
            Suffixes::ChangeDivisor,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_operand, change_divisor] = suffixes.try_into().unwrap();

        prefixes[Prefixes::RightOperand] * one
            + right_operand
            + prefixes[Prefixes::ChangeDivisor] * change_divisor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, VirtualChangeDivisorTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, VirtualChangeDivisorTable>();
    }
}
