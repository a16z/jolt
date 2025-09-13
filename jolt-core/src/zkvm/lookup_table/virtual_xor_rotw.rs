use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualXORROTWTable<const XLEN: usize, const ROTATION: u32>;

impl<const XLEN: usize, const ROTATION: u32> JoltLookupTable
    for VirtualXORROTWTable<XLEN, ROTATION>
{
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let x_32 = x as u32;
        let y_32 = y as u32;
        let xor_result = x_32 ^ y_32;
        xor_result.rotate_right(ROTATION) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut result = F::zero();
        // Process r in pairs, but only for the lower 32 bits (skip first XLEN/2 pairs)
        for (idx, chunk) in r.chunks_exact(2).enumerate().skip(XLEN / 2) {
            let r_x = chunk[0];
            let r_y = chunk[1];
            let xor_bit = (F::one() - r_x) * r_y + r_x * (F::one() - r_y);
            let position = idx - (XLEN / 2);
            let mut rotated_position = (position + ROTATION as usize) % 32;
            rotated_position = 31 - rotated_position;
            result += F::from_u64(1u64 << rotated_position) * xor_bit;
        }
        result
    }
}

impl<const XLEN: usize, const ROTATION: u32> PrefixSuffixDecomposition<XLEN>
    for VirtualXORROTWTable<XLEN, ROTATION>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        match ROTATION {
            7 => vec![Suffixes::One, Suffixes::XorRotW7],
            8 => vec![Suffixes::One, Suffixes::XorRotW8],
            12 => vec![Suffixes::One, Suffixes::XorRotW12],
            16 => vec![Suffixes::One, Suffixes::XorRotW16],
            _ => unimplemented!(),
        }
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor_rot] = suffixes.try_into().unwrap();
        match ROTATION {
            7 => prefixes[Prefixes::XorRotW7] * one + xor_rot,
            8 => prefixes[Prefixes::XorRotW8] * one + xor_rot,
            12 => prefixes[Prefixes::XorRotW12] * one + xor_rot,
            16 => prefixes[Prefixes::XorRotW16] * one + xor_rot,
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_64_xlen_test, lookup_table_mle_random_test,
        prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::VirtualXORROTWTable;

    type VirtualXORROTW7Table<const XLEN: usize> = VirtualXORROTWTable<XLEN, 7>;
    type VirtualXORROTW8Table<const XLEN: usize> = VirtualXORROTWTable<XLEN, 8>;
    type VirtualXORROTW12Table<const XLEN: usize> = VirtualXORROTWTable<XLEN, 12>;
    type VirtualXORROTW16Table<const XLEN: usize> = VirtualXORROTWTable<XLEN, 16>;

    #[test]
    fn prefix_suffix_7() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTW7Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_7() {
        lookup_table_mle_full_hypercube_64_xlen_test::<Fr, VirtualXORROTW7Table<XLEN>>();
    }

    #[test]
    fn mle_random_7() {
        lookup_table_mle_random_test::<Fr, VirtualXORROTW7Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix_8() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTW8Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_8() {
        lookup_table_mle_full_hypercube_64_xlen_test::<Fr, VirtualXORROTW8Table<XLEN>>();
    }

    #[test]
    fn mle_random_8() {
        lookup_table_mle_random_test::<Fr, VirtualXORROTW8Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix_12() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTW12Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_12() {
        lookup_table_mle_full_hypercube_64_xlen_test::<Fr, VirtualXORROTW12Table<XLEN>>();
    }

    #[test]
    fn mle_random_12() {
        lookup_table_mle_random_test::<Fr, VirtualXORROTW12Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix_16() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTW16Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_16() {
        lookup_table_mle_full_hypercube_64_xlen_test::<Fr, VirtualXORROTW16Table<XLEN>>();
    }

    #[test]
    fn mle_random_16() {
        lookup_table_mle_random_test::<Fr, VirtualXORROTW16Table<XLEN>>();
    }
}
