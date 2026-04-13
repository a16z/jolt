use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct EqualTable;

impl LookupTable for EqualTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x == y).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert!(r.len().is_multiple_of(2));
        let mut result = F::one();
        for i in (0..r.len()).step_by(2) {
            let x_i = r[i];
            let y_i = r[i + 1];
            result *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }
        result
    }
}

impl PrefixSuffixDecomposition for EqualTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::Eq]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [eq] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Eq] * eq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, EqualTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, EqualTable>();
    }
}
