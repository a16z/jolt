use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualChangeDivisorTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for VirtualChangeDivisorTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (dividend, divisor) = uninterleave_bits(index);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let dividend = dividend & mask;
        let divisor = divisor & mask;
        let signed_min = 1u64 << (XLEN - 1);
        let neg_one = mask;
        if dividend == signed_min && divisor == neg_one {
            1
        } else {
            divisor
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

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualChangeDivisorTable<XLEN> {
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
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, VirtualChangeDivisorTable<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, VirtualChangeDivisorTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualChangeDivisorTable<XLEN>>();
    }
}
