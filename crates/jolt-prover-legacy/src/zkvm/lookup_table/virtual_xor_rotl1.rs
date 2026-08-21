use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltLookupTable, PrefixSuffixDecomposition};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualXORROTL1Table<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for VirtualXORROTL1Table<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match XLEN {
            #[cfg(test)]
            8 => ((x as u8) ^ (y as u8).rotate_left(1)) as u64,
            64 => x ^ y.rotate_left(1),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_prev = r[2 * ((i + 1) % XLEN) + 1];
            result += F::from_u64(1 << (XLEN - 1 - i))
                * ((F::one() - x_i) * y_prev + x_i * (F::one() - y_prev));
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualXORROTL1Table<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::XorRotL1Pairs,
            Suffixes::TopYBit,
            Suffixes::BottomXBit,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, pairs, top_y, bottom_x] = suffixes.try_into().unwrap();
        prefixes[Prefixes::XorRotL1Acc] * one
            + pairs
            + prefixes[Prefixes::XorRotL1Straddle] * top_y
            + prefixes[Prefixes::XorRotL1Wrap] * bottom_x
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use common::constants::XLEN;

    use super::VirtualXORROTL1Table;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        prefix_suffix_test_with_phase_size,
    };

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualXORROTL1Table<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualXORROTL1Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTL1Table<XLEN>>();
        prefix_suffix_test_with_phase_size::<XLEN, Fr, VirtualXORROTL1Table<XLEN>>(8, 300);
    }
}
