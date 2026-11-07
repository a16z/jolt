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
            3 => &[Prefixes::XorRot3],
            4 => &[Prefixes::XorRot4],
            6 => &[Prefixes::XorRot6],
            9 => &[Prefixes::XorRot9],
            10 => &[Prefixes::XorRot10],
            14 => &[Prefixes::XorRot14],
            18 => &[Prefixes::XorRot18],
            21 => &[Prefixes::XorRot21],
            22 => &[Prefixes::XorRot22],
            23 => &[Prefixes::XorRot23],
            26 => &[Prefixes::XorRot26],
            27 => &[Prefixes::XorRot27],
            33 => &[Prefixes::XorRot33],
            43 => &[Prefixes::XorRot43],
            44 => &[Prefixes::XorRot44],
            45 => &[Prefixes::XorRot45],
            47 => &[Prefixes::XorRot47],
            49 => &[Prefixes::XorRot49],
            52 => &[Prefixes::XorRot52],
            60 => &[Prefixes::XorRot60],
            61 => &[Prefixes::XorRot61],
            32 => &[Prefixes::XorRot32],
            63 => &[Prefixes::XorRot63],
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        debug_assert_eq!(XLEN, 64);
        match ROTATION {
            16 => &[Suffixes::One, Suffixes::XorRot16],
            24 => &[Suffixes::One, Suffixes::XorRot24],
            3 => &[Suffixes::One, Suffixes::XorRot3],
            4 => &[Suffixes::One, Suffixes::XorRot4],
            6 => &[Suffixes::One, Suffixes::XorRot6],
            9 => &[Suffixes::One, Suffixes::XorRot9],
            10 => &[Suffixes::One, Suffixes::XorRot10],
            14 => &[Suffixes::One, Suffixes::XorRot14],
            18 => &[Suffixes::One, Suffixes::XorRot18],
            21 => &[Suffixes::One, Suffixes::XorRot21],
            22 => &[Suffixes::One, Suffixes::XorRot22],
            23 => &[Suffixes::One, Suffixes::XorRot23],
            26 => &[Suffixes::One, Suffixes::XorRot26],
            27 => &[Suffixes::One, Suffixes::XorRot27],
            33 => &[Suffixes::One, Suffixes::XorRot33],
            43 => &[Suffixes::One, Suffixes::XorRot43],
            44 => &[Suffixes::One, Suffixes::XorRot44],
            45 => &[Suffixes::One, Suffixes::XorRot45],
            47 => &[Suffixes::One, Suffixes::XorRot47],
            49 => &[Suffixes::One, Suffixes::XorRot49],
            52 => &[Suffixes::One, Suffixes::XorRot52],
            60 => &[Suffixes::One, Suffixes::XorRot60],
            61 => &[Suffixes::One, Suffixes::XorRot61],
            32 => &[Suffixes::One, Suffixes::XorRot32],
            63 => &[Suffixes::One, Suffixes::XorRot63],
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
            3 => prefixes[Prefixes::XorRot3] * one + xor_rot,
            4 => prefixes[Prefixes::XorRot4] * one + xor_rot,
            6 => prefixes[Prefixes::XorRot6] * one + xor_rot,
            9 => prefixes[Prefixes::XorRot9] * one + xor_rot,
            10 => prefixes[Prefixes::XorRot10] * one + xor_rot,
            14 => prefixes[Prefixes::XorRot14] * one + xor_rot,
            18 => prefixes[Prefixes::XorRot18] * one + xor_rot,
            21 => prefixes[Prefixes::XorRot21] * one + xor_rot,
            22 => prefixes[Prefixes::XorRot22] * one + xor_rot,
            23 => prefixes[Prefixes::XorRot23] * one + xor_rot,
            26 => prefixes[Prefixes::XorRot26] * one + xor_rot,
            27 => prefixes[Prefixes::XorRot27] * one + xor_rot,
            33 => prefixes[Prefixes::XorRot33] * one + xor_rot,
            43 => prefixes[Prefixes::XorRot43] * one + xor_rot,
            44 => prefixes[Prefixes::XorRot44] * one + xor_rot,
            45 => prefixes[Prefixes::XorRot45] * one + xor_rot,
            47 => prefixes[Prefixes::XorRot47] * one + xor_rot,
            49 => prefixes[Prefixes::XorRot49] * one + xor_rot,
            52 => prefixes[Prefixes::XorRot52] * one + xor_rot,
            60 => prefixes[Prefixes::XorRot60] * one + xor_rot,
            61 => prefixes[Prefixes::XorRot61] * one + xor_rot,
            32 => prefixes[Prefixes::XorRot32] * one + xor_rot,
            63 => prefixes[Prefixes::XorRot63] * one + xor_rot,
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
    fn keccak_rotation_tables_match_their_boolean_values_and_phase_decomposition() {
        macro_rules! check {
            ($($rotation:literal),* $(,)?) => { $(
                mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, $rotation>>();
                prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, $rotation>>();
            )* };
        }
        check!(3, 4, 6, 9, 10, 14, 18, 21, 22, 23, 26, 27, 33, 43, 44, 45, 47, 49, 52, 60, 61);
    }

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

    #[test]
    fn mle_full_hypercube_rot32() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 32>>();
    }

    #[test]
    fn mle_full_hypercube_rot63() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 63>>();
    }
}
