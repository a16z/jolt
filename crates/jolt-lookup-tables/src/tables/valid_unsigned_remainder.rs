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
pub struct ValidUnsignedRemainderTable;

impl LookupTable for ValidUnsignedRemainderTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (remainder, divisor) = uninterleave_bits(index);
        (divisor == 0 || remainder < divisor).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let mut divisor_is_zero = F::one();
        let mut lt = F::zero();
        let mut eq = F::one();

        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            divisor_is_zero *= F::one() - y_i;
            lt += (F::one() - x_i) * y_i * eq;
            eq *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }

        lt + divisor_is_zero
    }
}

impl PrefixSuffixDecomposition for ValidUnsignedRemainderTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::LessThan,
            Suffixes::RightOperandIsZero,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than, right_operand_is_zero] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightOperandIsZero] * right_operand_is_zero
            + prefixes[Prefixes::LessThan] * one
            + prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, ValidUnsignedRemainderTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, ValidUnsignedRemainderTable>();
    }
}
