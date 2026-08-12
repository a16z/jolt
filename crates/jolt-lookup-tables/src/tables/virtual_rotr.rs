use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::{LookupMaterializer, MaterializerBackend, U128Materializer};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualROTRTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for VirtualROTRTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        self.materialize(&mut U128Materializer::<XLEN>::new(index))
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: crate::LookupEval + FieldOps<C>,
    {
        assert_eq!(r.len() % 2, 0, "r must have even length");
        assert_eq!(r.len() / 2, XLEN, "r must have length 2 * XLEN");

        let mut prod_one_plus_y = F::one();
        let mut first_sum = F::zero();
        let mut second_sum = F::zero();

        for (i, chunk) in r.chunks_exact(2).enumerate() {
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

impl<const XLEN: usize> LookupMaterializer<XLEN> for VirtualROTRTable<XLEN> {
    fn materialize<B: MaterializerBackend>(&self, backend: &mut B) -> B::Output {
        let zero = backend.nat_constant(0);
        let one = backend.nat_constant(1);
        let mut prod_one_plus_y = one.clone();
        let mut first_sum = zero.clone();
        let mut second_sum = zero;

        for i in 0..XLEN {
            let x_bit = backend.input_bit(2 * i);
            let y_bit = backend.input_bit(2 * i + 1);
            let not_y_bit = backend.not(y_bit.clone());
            let x = backend.bit_to_nat(x_bit);
            let y = backend.bit_to_nat(y_bit);
            let one_minus_y = backend.bit_to_nat(not_y_bit);

            let one_plus_y = backend.nat_add(one.clone(), y.clone());
            let scaled_first_sum = backend.nat_mul(first_sum, one_plus_y.clone());
            let xy = backend.nat_mul(x.clone(), y);
            first_sum = backend.nat_add(scaled_first_sum, xy);

            let zero_selected_x = backend.nat_mul(x, one_minus_y);
            let shifted_zero_selected_x = backend.nat_mul(zero_selected_x, prod_one_plus_y.clone());
            let weight = backend.nat_constant(1u128 << (XLEN - 1 - i));
            let weighted_zero_selected_x = backend.nat_mul(shifted_zero_selected_x, weight);
            second_sum = backend.nat_add(second_sum, weighted_zero_selected_x);

            prod_one_plus_y = backend.nat_mul(prod_one_plus_y, one_plus_y);
        }

        let value = backend.nat_add(first_sum, second_sum);
        backend.output(value)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualROTRTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[
            Prefixes::RightShift,
            Prefixes::LeftShiftHelper,
            Prefixes::LeftShift,
        ]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::RightShiftHelper,
            Suffixes::RightShift,
            Suffixes::LeftShift,
            Suffixes::One,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [right_shift_helper, right_shift, left_shift, one] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * right_shift_helper
            + right_shift
            + prefixes[Prefixes::LeftShiftHelper] * left_shift
            + prefixes[Prefixes::LeftShift] * one
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        crate::tables::test_utils::gen_bitmask_lookup_index::<XLEN>(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::{uninterleave_bits, XLEN};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, VirtualROTRTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualROTRTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, VirtualROTRTable<8>>();
    }

    #[test]
    fn shared_materializer_matches_original_recurrence() {
        const TEST_XLEN: usize = 8;
        for index in 0..(1u128 << (2 * TEST_XLEN)) {
            let (x_bits, y_bits) = uninterleave_bits(index);
            let mut prod_one_plus_y = 1u128;
            let mut first_sum = 0u64;
            let mut second_sum = 0u64;

            for i in (0..TEST_XLEN).rev() {
                let x = x_bits >> i & 1;
                let y = y_bits >> i & 1;
                first_sum = first_sum * (1 + y) + x * y;
                second_sum += x * ((1 - u128::from(y)) * prod_one_plus_y) as u64 * (1 << i);
                prod_one_plus_y *= 1 + u128::from(y);
            }

            assert_eq!(
                VirtualROTRTable::<TEST_XLEN>.materialize_entry(index),
                first_sum + second_sum
            );
        }
    }
}
