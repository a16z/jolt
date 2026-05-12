use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::signed_less_than::SignedLessThanTable;
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SignedGreaterThanEqualTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for SignedGreaterThanEqualTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        // Sign-extend the lower XLEN bits to a full i64 before comparing.
        let shift = 64 - XLEN;
        let x_signed = ((x as i64) << shift) >> shift;
        let y_signed = ((y as i64) << shift) >> shift;
        (x_signed >= y_signed).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        F::one() - SignedLessThanTable::<XLEN>.evaluate_mle(r)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for SignedGreaterThanEqualTable<XLEN> {
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
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, SignedGreaterThanEqualTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, SignedGreaterThanEqualTable<8>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, SignedGreaterThanEqualTable<XLEN>>();
    }
}
