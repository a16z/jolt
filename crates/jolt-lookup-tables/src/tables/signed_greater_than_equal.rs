use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::signed_less_than::SignedLessThanTable;
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignedGreaterThanEqualTable;

impl LookupTable for SignedGreaterThanEqualTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x as i64 >= y as i64).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        F::one() - SignedLessThanTable.evaluate_mle(r)
    }
}

impl PrefixSuffixDecomposition for SignedGreaterThanEqualTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::LessThan]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        // 1 - LT(x, y) = 1 - (isNegative(x) && isPositive(y)) - LTU(x, y)
        one + prefixes[Prefixes::RightOperandMsb] * one
            - prefixes[Prefixes::LeftOperandMsb] * one
            - prefixes[Prefixes::LessThan] * one
            - prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, SignedGreaterThanEqualTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, SignedGreaterThanEqualTable>();
    }
}
