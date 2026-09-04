use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualXORROTTable<const XLEN: usize, const ROTATION: u32>;

impl<const XLEN: usize, const ROTATION: u32> LookupTable for VirtualXORROTTable<XLEN, ROTATION> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let xor_result = x ^ y;
        let r = (ROTATION as usize) % XLEN;
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let v = (xor_result & mask) as u128;
        (((v >> r) | (v << (XLEN - r))) as u64) & mask
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
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
    fn prefixes(&self) -> &'static [Prefixes] {
        match ROTATION {
            16 => &[Prefixes::XorRot16],
            24 => &[Prefixes::XorRot24],
            32 => &[Prefixes::XorRot32],
            63 => &[Prefixes::XorRot63],
            2 => &[Prefixes::XorRot2],
            3 => &[Prefixes::XorRot3],
            8 => &[Prefixes::XorRot8],
            9 => &[Prefixes::XorRot9],
            19 => &[Prefixes::XorRot19],
            20 => &[Prefixes::XorRot20],
            21 => &[Prefixes::XorRot21],
            23 => &[Prefixes::XorRot23],
            25 => &[Prefixes::XorRot25],
            28 => &[Prefixes::XorRot28],
            36 => &[Prefixes::XorRot36],
            37 => &[Prefixes::XorRot37],
            39 => &[Prefixes::XorRot39],
            43 => &[Prefixes::XorRot43],
            44 => &[Prefixes::XorRot44],
            46 => &[Prefixes::XorRot46],
            49 => &[Prefixes::XorRot49],
            50 => &[Prefixes::XorRot50],
            54 => &[Prefixes::XorRot54],
            56 => &[Prefixes::XorRot56],
            58 => &[Prefixes::XorRot58],
            61 => &[Prefixes::XorRot61],
            62 => &[Prefixes::XorRot62],
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        debug_assert_eq!(XLEN, 64);
        match ROTATION {
            16 => &[Suffixes::One, Suffixes::XorRot16],
            24 => &[Suffixes::One, Suffixes::XorRot24],
            32 => &[Suffixes::One, Suffixes::XorRot32],
            63 => &[Suffixes::One, Suffixes::XorRot63],
            2 => &[Suffixes::One, Suffixes::XorRot2],
            3 => &[Suffixes::One, Suffixes::XorRot3],
            8 => &[Suffixes::One, Suffixes::XorRot8],
            9 => &[Suffixes::One, Suffixes::XorRot9],
            19 => &[Suffixes::One, Suffixes::XorRot19],
            20 => &[Suffixes::One, Suffixes::XorRot20],
            21 => &[Suffixes::One, Suffixes::XorRot21],
            23 => &[Suffixes::One, Suffixes::XorRot23],
            25 => &[Suffixes::One, Suffixes::XorRot25],
            28 => &[Suffixes::One, Suffixes::XorRot28],
            36 => &[Suffixes::One, Suffixes::XorRot36],
            37 => &[Suffixes::One, Suffixes::XorRot37],
            39 => &[Suffixes::One, Suffixes::XorRot39],
            43 => &[Suffixes::One, Suffixes::XorRot43],
            44 => &[Suffixes::One, Suffixes::XorRot44],
            46 => &[Suffixes::One, Suffixes::XorRot46],
            49 => &[Suffixes::One, Suffixes::XorRot49],
            50 => &[Suffixes::One, Suffixes::XorRot50],
            54 => &[Suffixes::One, Suffixes::XorRot54],
            56 => &[Suffixes::One, Suffixes::XorRot56],
            58 => &[Suffixes::One, Suffixes::XorRot58],
            61 => &[Suffixes::One, Suffixes::XorRot61],
            62 => &[Suffixes::One, Suffixes::XorRot62],
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }

    #[expect(clippy::unwrap_used)]
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
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random_rot32() {
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 32>>();
    }

    #[test]
    fn prefix_suffix_rot32() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 32>>();
    }

    #[test]
    fn mle_random_rot24() {
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 24>>();
    }

    #[test]
    fn prefix_suffix_rot24() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 24>>();
    }

    #[test]
    fn mle_random_rot16() {
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 16>>();
    }

    #[test]
    fn prefix_suffix_rot16() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 16>>();
    }

    #[test]
    fn mle_random_rot63() {
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 63>>();
    }

    #[test]
    fn prefix_suffix_rot63() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 63>>();
    }

    macro_rules! keccak_xor_rot_tests {
        ($($n:literal => ($random:ident, $ps:ident)),+ $(,)?) => {
            $(
                #[test]
                fn $random() {
                    mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, $n>>();
                }

                #[test]
                fn $ps() {
                    prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, $n>>();
                }
            )+
        };
    }

    keccak_xor_rot_tests!(
        2 => (mle_random_rot2, prefix_suffix_rot2),
        3 => (mle_random_rot3, prefix_suffix_rot3),
        8 => (mle_random_rot8, prefix_suffix_rot8),
        9 => (mle_random_rot9, prefix_suffix_rot9),
        19 => (mle_random_rot19, prefix_suffix_rot19),
        20 => (mle_random_rot20, prefix_suffix_rot20),
        21 => (mle_random_rot21, prefix_suffix_rot21),
        23 => (mle_random_rot23, prefix_suffix_rot23),
        25 => (mle_random_rot25, prefix_suffix_rot25),
        28 => (mle_random_rot28, prefix_suffix_rot28),
        36 => (mle_random_rot36, prefix_suffix_rot36),
        37 => (mle_random_rot37, prefix_suffix_rot37),
        39 => (mle_random_rot39, prefix_suffix_rot39),
        43 => (mle_random_rot43, prefix_suffix_rot43),
        44 => (mle_random_rot44, prefix_suffix_rot44),
        46 => (mle_random_rot46, prefix_suffix_rot46),
        49 => (mle_random_rot49, prefix_suffix_rot49),
        50 => (mle_random_rot50, prefix_suffix_rot50),
        54 => (mle_random_rot54, prefix_suffix_rot54),
        56 => (mle_random_rot56, prefix_suffix_rot56),
        58 => (mle_random_rot58, prefix_suffix_rot58),
        61 => (mle_random_rot61, prefix_suffix_rot61),
        62 => (mle_random_rot62, prefix_suffix_rot62),
    );

    #[test]
    fn mle_full_hypercube_rot16() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 16>>();
    }

    #[test]
    fn mle_full_hypercube_rot24() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 24>>();
    }

    #[test]
    fn mle_full_hypercube_rot32() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 32>>();
    }

    #[test]
    fn mle_full_hypercube_rot63() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 63>>();
    }
}
