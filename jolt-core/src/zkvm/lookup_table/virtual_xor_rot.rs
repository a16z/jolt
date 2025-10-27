use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable,
    PrefixSuffixDecomposition,
};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::uninterleave_bits,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualXORROTTable<const XLEN: usize, const ROTATION: u32>;

impl<const XLEN: usize, const ROTATION: u32> JoltLookupTable
    for VirtualXORROTTable<XLEN, ROTATION>
{
    fn materialize_entry(&self, index: u128) -> u64 {
        match XLEN {
            #[cfg(test)]
            8 => {
                let (x, y) = uninterleave_bits(index);
                let xor_result = x as u8 ^ y as u8;
                xor_result.rotate_right(ROTATION) as u64
            }
            64 => {
                let (x, y) = uninterleave_bits(index);
                let xor_result = x ^ y;
                xor_result.rotate_right(ROTATION)
            }
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
            let y_i = r[2 * i + 1];

            let rotated_position = (i + ROTATION as usize) % XLEN;
            let bit_position = XLEN - 1 - rotated_position;

            result += F::from_u64(1u64 << bit_position)
                * ((F::one() - x_i) * y_i + x_i * (F::one() - y_i));
        }
        result
    }
}

impl<const XLEN: usize, const ROTATION: u32> PrefixSuffixDecomposition<XLEN>
    for VirtualXORROTTable<XLEN, ROTATION>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        debug_assert_eq!(XLEN, 64);
        match ROTATION {
            16 => vec![Suffixes::One, Suffixes::XorRot16],
            24 => vec![Suffixes::One, Suffixes::XorRot24],
            32 => vec![Suffixes::One, Suffixes::XorRot32],
            63 => vec![Suffixes::One, Suffixes::XorRot63],
            _ => unimplemented!(),
        }
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor_rot] = suffixes.try_into().unwrap();
        match ROTATION {
            16 => prefixes[Prefixes::XorRot16] * one + xor_rot,
            24 => prefixes[Prefixes::XorRot24] * one + xor_rot,
            32 => prefixes[Prefixes::XorRot32] * one + xor_rot,
            63 => prefixes[Prefixes::XorRot63] * one + xor_rot,
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use common::constants::XLEN;

    use super::VirtualXORROTTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test,
        lookup_table_mle_random_test,
        prefix_suffix_test,
    };

    // Type aliases for different rotation amounts
    type VirtualXORROT16Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 16>;
    type VirtualXORROT24Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 24>;
    type VirtualXORROT32Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 32>;
    type VirtualXORROT63Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 63>;

    // Tests for rotation by 16
    #[test]
    fn prefix_suffix_16() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROT16Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_16() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualXORROT16Table<8>>();
    }

    #[test]
    fn mle_random_16() {
        lookup_table_mle_random_test::<Fr, VirtualXORROT16Table<XLEN>>();
    }

    // Tests for rotation by 24
    #[test]
    fn prefix_suffix_24() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROT24Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_24() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualXORROT24Table<8>>();
    }

    #[test]
    fn mle_random_24() {
        lookup_table_mle_random_test::<Fr, VirtualXORROT24Table<XLEN>>();
    }

    // Tests for rotation by 32
    #[test]
    fn prefix_suffix_32() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROT32Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_32() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualXORROT32Table<8>>();
    }

    #[test]
    fn mle_random_32() {
        lookup_table_mle_random_test::<Fr, VirtualXORROT32Table<XLEN>>();
    }

    // Tests for rotation by 63
    #[test]
    fn prefix_suffix_63() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROT63Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube_63() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualXORROT63Table<8>>();
    }

    #[test]
    fn mle_random_63() {
        lookup_table_mle_random_test::<Fr, VirtualXORROT63Table<XLEN>>();
    }
}
