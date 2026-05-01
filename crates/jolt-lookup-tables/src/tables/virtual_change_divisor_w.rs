use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualChangeDivisorWTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for VirtualChangeDivisorWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (dividend, divisor) = uninterleave_bits(index);
        let half = XLEN / 2;
        let half_mask = (1u128 << half).wrapping_sub(1) as u64;
        let dividend_lo = dividend & half_mask;
        let divisor_lo = divisor & half_mask;
        let signed_min = 1u64 << (half - 1);
        let neg_one_half = half_mask;
        if dividend_lo == signed_min && divisor_lo == neg_one_half {
            1
        } else {
            let sign_bit = (divisor_lo >> (half - 1)) & 1;
            if sign_bit == 1 {
                let upper = ((1u128 << XLEN).wrapping_sub(1) as u64) ^ half_mask;
                divisor_lo | upper
            } else {
                divisor_lo
            }
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let sign_bit = r[XLEN + 1];

        let mut divisor_value = F::zero();
        for i in XLEN / 2..XLEN {
            let bit_value = r[2 * i + 1];
            let shift = XLEN - 1 - i;
            divisor_value += F::from_u64(1u64 << shift) * bit_value;
        }

        let mut x_product = r[XLEN].into();
        for i in XLEN / 2 + 1..XLEN {
            x_product *= F::one() - r[2 * i];
        }

        let mut y_product = F::one();
        for i in XLEN / 2..XLEN {
            y_product = y_product * r[2 * i + 1];
        }

        let sign_extension = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))) * sign_bit;
        let adjustment = F::from_u64(2) - F::from_u128(1u128 << XLEN);

        divisor_value + adjustment * x_product * y_product + sign_extension
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualChangeDivisorWTable<XLEN> {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::RightOperandW,
            Suffixes::ChangeDivisorW,
            Suffixes::SignExtensionRightOperand,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_operand_w, change_divisor_w, sign_extension] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightOperandW] * one
            + right_operand_w
            + prefixes[Prefixes::ChangeDivisorW] * change_divisor_w
            + prefixes[Prefixes::SignExtensionRightOperand] * sign_extension
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
        mle_full_hypercube_test::<8, Fr, VirtualChangeDivisorWTable<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, VirtualChangeDivisorWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualChangeDivisorWTable<XLEN>>();
    }
}
