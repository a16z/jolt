use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{JoltField, MontU128};
use crate::utils::uninterleave_bits;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualRotrTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for VirtualRotrTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x_bits, y_bits) = uninterleave_bits(index);

        let mut prod_one_plus_y = 1;
        let mut first_sum = 0;
        let mut second_sum = 0;

        (0..WORD_SIZE).rev().for_each(|i| {
            let x = x_bits >> i & 1;
            let y = y_bits >> i & 1;
            first_sum *= 1 + y;
            first_sum += x * y;
            second_sum += x * (1 - y) * prod_one_plus_y * (1 << i);
            prod_one_plus_y *= 1 + y;
        });

        (first_sum + second_sum) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        assert_eq!(r.len() % 2, 0, "r must have even length");
        assert_eq!(r.len() / 2, WORD_SIZE, "r must have length 2 * WORD_SIZE");

        let mut prod_one_plus_y = F::one();
        let mut first_sum = F::zero();
        let mut second_sum = F::zero();

        // Process r in pairs (r_x, r_y)
        for (i, chunk) in r.chunks_exact(2).enumerate() {
            let r_x = chunk[0];
            let r_y = chunk[1];

            // Update first_sum: multiply by (1 + r_y) then add r_x * r_y
            first_sum *= F::one() + F::from_u128_mont(r_y);
            first_sum += F::from_u128_mont(r_x) * F::from_u128_mont(r_y);

            // Update second_sum
            second_sum +=
                (F::one() - F::from_u128_mont(r_y)).mul_u128_mont_form(r_x) * prod_one_plus_y * F::from_u64(1 << (WORD_SIZE - 1 - i));

            // Update prod_one_plus_y for next iteration
            prod_one_plus_y *= F::one() + F::from_u128_mont(r_y);
        }

        first_sum + second_sum
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for VirtualRotrTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::RightShiftHelper,
            Suffixes::RightShift,
            Suffixes::LeftShift,
            Suffixes::One,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [right_shift_helper, right_shift, left_shift, one] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * right_shift_helper
            + right_shift
            + prefixes[Prefixes::LeftShiftHelper] * left_shift
            + prefixes[Prefixes::LeftShift] * one
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u64 {
        super::test::gen_bitmask_lookup_index(rng)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::VirtualRotrTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualRotrTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualRotrTable<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, VirtualRotrTable<32>>();
    }
}
