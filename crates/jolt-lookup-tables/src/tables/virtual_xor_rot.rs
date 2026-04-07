use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;
use crate::XLEN;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualXORROTTable<const ROTATION: u32>;

impl<const ROTATION: u32> LookupTable for VirtualXORROTTable<ROTATION> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let xor_result = x ^ y;
        xor_result.rotate_right(ROTATION)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
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

impl<const ROTATION: u32> PrefixSuffixDecomposition for VirtualXORROTTable<ROTATION> {
    fn suffixes(&self) -> &'static [Suffixes] {
        debug_assert_eq!(XLEN, 64);
        match ROTATION {
            16 => &[Suffixes::One, Suffixes::XorRot16],
            24 => &[Suffixes::One, Suffixes::XorRot24],
            32 => &[Suffixes::One, Suffixes::XorRot32],
            63 => &[Suffixes::One, Suffixes::XorRot63],
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor_rot] = suffixes.try_into().unwrap();
        match ROTATION {
            16 => prefixes[Prefixes::XorRot16] * one + xor_rot,
            24 => prefixes[Prefixes::XorRot24] * one + xor_rot,
            32 => prefixes[Prefixes::XorRot32] * one + xor_rot,
            63 => prefixes[Prefixes::XorRot63] * one + xor_rot,
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random_rot32() {
        mle_random_test::<Fr, VirtualXORROTTable<32>>();
    }

    #[test]
    fn prefix_suffix_rot32() {
        prefix_suffix_test::<Fr, VirtualXORROTTable<32>>();
    }

    #[test]
    fn mle_random_rot24() {
        mle_random_test::<Fr, VirtualXORROTTable<24>>();
    }

    #[test]
    fn prefix_suffix_rot24() {
        prefix_suffix_test::<Fr, VirtualXORROTTable<24>>();
    }

    #[test]
    fn mle_random_rot16() {
        mle_random_test::<Fr, VirtualXORROTTable<16>>();
    }

    #[test]
    fn prefix_suffix_rot16() {
        prefix_suffix_test::<Fr, VirtualXORROTTable<16>>();
    }

    #[test]
    fn mle_random_rot63() {
        mle_random_test::<Fr, VirtualXORROTTable<63>>();
    }

    #[test]
    fn prefix_suffix_rot63() {
        prefix_suffix_test::<Fr, VirtualXORROTTable<63>>();
    }
}
