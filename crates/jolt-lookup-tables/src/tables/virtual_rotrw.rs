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
pub struct VirtualROTRWTable;

impl LookupTable for VirtualROTRWTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x_bits, y_bits) = uninterleave_bits(index);

        let mut prod_one_plus_y = 1;
        let mut first_sum = 0;
        let mut second_sum = 0;

        (0..XLEN).rev().skip(XLEN / 2).for_each(|i| {
            let x = x_bits >> i & 1;
            let y = y_bits >> i & 1;
            first_sum *= 1 + y;
            first_sum += x * y;
            second_sum += x * (1 - y) * prod_one_plus_y * (1 << i);
            prod_one_plus_y *= 1 + y;
        });

        first_sum + second_sum
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        assert_eq!(r.len() % 2, 0, "r must have even length");
        assert_eq!(r.len() / 2, XLEN, "r must have length 2 * XLEN");

        let mut prod_one_plus_y = F::one();
        let mut first_sum = F::zero();
        let mut second_sum = F::zero();

        for (i, chunk) in r.chunks_exact(2).enumerate().skip(XLEN / 2) {
            let r_x = chunk[0];
            let r_y = chunk[1];

            first_sum *= F::one() + r_y;
            first_sum += r_x * r_y;

            second_sum +=
                r_x * (F::one() - r_y) * prod_one_plus_y * F::from_u64(1 << (XLEN - 1 - i));

            prod_one_plus_y *= F::one() + r_y;
        }

        first_sum + second_sum
    }
}

impl PrefixSuffixDecomposition for VirtualROTRWTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::RightShiftWHelper,
            Suffixes::RightShiftW,
            Suffixes::LeftShiftW,
            Suffixes::One,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [right_shift_w_helper, right_shift_w, left_shift_w, one] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShiftW] * right_shift_w_helper
            + right_shift_w
            + prefixes[Prefixes::LeftShiftWHelper] * left_shift_w
            + prefixes[Prefixes::LeftShiftW] * one
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        crate::tables::test_utils::gen_bitmask_lookup_index(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, VirtualROTRWTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, VirtualROTRWTable>();
    }
}
