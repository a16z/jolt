use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualRotrWTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for VirtualRotrWTable<XLEN> {
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

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        assert_eq!(r.len() % 2, 0, "r must have even length");
        assert_eq!(r.len() / 2, XLEN, "r must have length 2 * XLEN");

        let mut prod_one_plus_y = F::one();
        let mut first_sum = F::zero();
        let mut second_sum = F::zero();

        // Process r in pairs (r_x, r_y)
        for (i, chunk) in r.chunks_exact(2).enumerate().skip(XLEN / 2) {
            let r_x = chunk[0];
            let r_y = chunk[1];

            // Update first_sum: multiply by (1 + r_y) then add r_x * r_y
            first_sum *= F::one() + r_y;
            first_sum += r_x * r_y;

            // Update second_sum
            second_sum +=
                r_x * (F::one() - r_y) * prod_one_plus_y * F::from_u64(1 << (XLEN - 1 - i));

            // Update prod_one_plus_y for next iteration
            prod_one_plus_y *= F::one() + r_y;
        }

        first_sum + second_sum
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualRotrWTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::RightShiftWHelper,
            Suffixes::RightShiftW,
            Suffixes::LeftShiftW,
            Suffixes::One,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [right_shift_w_helper, right_shift_w, left_shift_w, one] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShiftW] * right_shift_w_helper
            + right_shift_w
            + prefixes[Prefixes::LeftShiftWHelper] * left_shift_w
            + prefixes[Prefixes::LeftShiftW] * one
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        super::test::gen_bitmask_lookup_index(rng)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::VirtualRotrWTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualRotrWTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualRotrWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualRotrWTable<XLEN>>();
    }
}
