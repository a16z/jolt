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
pub struct UnsignedLessThanTable;

impl LookupTable for UnsignedLessThanTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x < y).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        // \sum_i (1 - x_i) * y_i * \prod_{j < i} ((1 - x_j) * (1 - y_j) + x_j * y_j)
        let mut result = F::zero();
        let mut eq_term = F::one();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += (F::one() - x_i) * y_i * eq_term;
            eq_term *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }
        result
    }
}

impl PrefixSuffixDecomposition for UnsignedLessThanTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::LessThan]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LessThan] * one + prefixes[Prefixes::Eq] * less_than
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, UnsignedLessThanTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, UnsignedLessThanTable>();
    }
}
