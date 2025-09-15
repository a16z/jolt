use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidUpperBits0<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for ValidUpperBits0<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        // Check if upper XLEN bits are all zero
        let upper_bits = index >> XLEN;
        if upper_bits == 0 {
            1  // Valid: no overflow
        } else {
            0  // Invalid: overflow occurred
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);
        
        // We want to return 1 if all upper bits are 0, otherwise 0
        // This is the product of (1 - r_i) for all upper bits
        let mut result = F::one();
        for i in 0..XLEN {
            // r[i] corresponds to the i-th bit of the upper word
            result *= F::one() - r[i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ValidUpperBits0<XLEN> {
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
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        super::test::gen_bitmask_lookup_index(rng)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::ValidUpperBits0;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ValidUpperBits0<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ValidUpperBits0<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ValidUpperBits0<XLEN>>();
    }
}
