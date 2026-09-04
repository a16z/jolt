use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::uninterleave_bits;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

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
            2 => vec![Suffixes::One, Suffixes::XorRot2],
            3 => vec![Suffixes::One, Suffixes::XorRot3],
            8 => vec![Suffixes::One, Suffixes::XorRot8],
            9 => vec![Suffixes::One, Suffixes::XorRot9],
            19 => vec![Suffixes::One, Suffixes::XorRot19],
            20 => vec![Suffixes::One, Suffixes::XorRot20],
            21 => vec![Suffixes::One, Suffixes::XorRot21],
            23 => vec![Suffixes::One, Suffixes::XorRot23],
            25 => vec![Suffixes::One, Suffixes::XorRot25],
            28 => vec![Suffixes::One, Suffixes::XorRot28],
            36 => vec![Suffixes::One, Suffixes::XorRot36],
            37 => vec![Suffixes::One, Suffixes::XorRot37],
            39 => vec![Suffixes::One, Suffixes::XorRot39],
            43 => vec![Suffixes::One, Suffixes::XorRot43],
            44 => vec![Suffixes::One, Suffixes::XorRot44],
            46 => vec![Suffixes::One, Suffixes::XorRot46],
            49 => vec![Suffixes::One, Suffixes::XorRot49],
            50 => vec![Suffixes::One, Suffixes::XorRot50],
            54 => vec![Suffixes::One, Suffixes::XorRot54],
            56 => vec![Suffixes::One, Suffixes::XorRot56],
            58 => vec![Suffixes::One, Suffixes::XorRot58],
            61 => vec![Suffixes::One, Suffixes::XorRot61],
            62 => vec![Suffixes::One, Suffixes::XorRot62],
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
            2 => prefixes[Prefixes::XorRot2] * one + xor_rot,
            3 => prefixes[Prefixes::XorRot3] * one + xor_rot,
            8 => prefixes[Prefixes::XorRot8] * one + xor_rot,
            9 => prefixes[Prefixes::XorRot9] * one + xor_rot,
            19 => prefixes[Prefixes::XorRot19] * one + xor_rot,
            20 => prefixes[Prefixes::XorRot20] * one + xor_rot,
            21 => prefixes[Prefixes::XorRot21] * one + xor_rot,
            23 => prefixes[Prefixes::XorRot23] * one + xor_rot,
            25 => prefixes[Prefixes::XorRot25] * one + xor_rot,
            28 => prefixes[Prefixes::XorRot28] * one + xor_rot,
            36 => prefixes[Prefixes::XorRot36] * one + xor_rot,
            37 => prefixes[Prefixes::XorRot37] * one + xor_rot,
            39 => prefixes[Prefixes::XorRot39] * one + xor_rot,
            43 => prefixes[Prefixes::XorRot43] * one + xor_rot,
            44 => prefixes[Prefixes::XorRot44] * one + xor_rot,
            46 => prefixes[Prefixes::XorRot46] * one + xor_rot,
            49 => prefixes[Prefixes::XorRot49] * one + xor_rot,
            50 => prefixes[Prefixes::XorRot50] * one + xor_rot,
            54 => prefixes[Prefixes::XorRot54] * one + xor_rot,
            56 => prefixes[Prefixes::XorRot56] * one + xor_rot,
            58 => prefixes[Prefixes::XorRot58] * one + xor_rot,
            61 => prefixes[Prefixes::XorRot61] * one + xor_rot,
            62 => prefixes[Prefixes::XorRot62] * one + xor_rot,
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::VirtualXORROTTable;

    // Type aliases for different rotation amounts
    type VirtualXORROT16Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 16>;
    type VirtualXORROT24Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 24>;
    type VirtualXORROT32Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 32>;
    type VirtualXORROT63Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 63>;

    macro_rules! keccak_xor_rot_tests {
        ($($n:literal => ($ps:ident, $random:ident)),+ $(,)?) => {
            $(
                #[test]
                fn $ps() {
                    prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, $n>>();
                }

                #[test]
                fn $random() {
                    lookup_table_mle_random_test::<Fr, VirtualXORROTTable<XLEN, $n>>();
                }
            )+
        };
    }

    keccak_xor_rot_tests!(
        2 => (prefix_suffix_2, mle_random_2),
        3 => (prefix_suffix_3, mle_random_3),
        8 => (prefix_suffix_8, mle_random_8),
        9 => (prefix_suffix_9, mle_random_9),
        19 => (prefix_suffix_19, mle_random_19),
        20 => (prefix_suffix_20, mle_random_20),
        21 => (prefix_suffix_21, mle_random_21),
        23 => (prefix_suffix_23, mle_random_23),
        25 => (prefix_suffix_25, mle_random_25),
        28 => (prefix_suffix_28, mle_random_28),
        36 => (prefix_suffix_36, mle_random_36),
        37 => (prefix_suffix_37, mle_random_37),
        39 => (prefix_suffix_39, mle_random_39),
        43 => (prefix_suffix_43, mle_random_43),
        44 => (prefix_suffix_44, mle_random_44),
        46 => (prefix_suffix_46, mle_random_46),
        49 => (prefix_suffix_49, mle_random_49),
        50 => (prefix_suffix_50, mle_random_50),
        54 => (prefix_suffix_54, mle_random_54),
        56 => (prefix_suffix_56, mle_random_56),
        58 => (prefix_suffix_58, mle_random_58),
        61 => (prefix_suffix_61, mle_random_61),
        62 => (prefix_suffix_62, mle_random_62),
    );

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
