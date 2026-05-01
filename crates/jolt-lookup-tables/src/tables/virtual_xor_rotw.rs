use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualXORROTWTable<const XLEN: usize, const ROTATION: u32>;

impl<const XLEN: usize, const ROTATION: u32> LookupTable for VirtualXORROTWTable<XLEN, ROTATION> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let half = XLEN / 2;
        let r = (ROTATION as usize) % half;
        let half_mask = (1u128 << half).wrapping_sub(1) as u64;
        let xor_result = ((x as u128 ^ y as u128) & half_mask as u128) as u64;
        let v = xor_result as u128;
        (((v >> r) | (v << (half - r))) as u64) & half_mask
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for (idx, chunk) in r.chunks_exact(2).enumerate().skip(XLEN / 2) {
            let r_x = chunk[0];
            let r_y = chunk[1];
            let xor_bit = (F::one() - r_x) * r_y + r_x * (F::one() - r_y);
            let position = idx - (XLEN / 2);
            let mut rotated_position = (position + ROTATION as usize) % (XLEN / 2);
            rotated_position = (XLEN / 2) - 1 - rotated_position;
            result += F::from_u64(1u64 << rotated_position) * xor_bit;
        }
        result
    }
}

impl<const XLEN: usize, const ROTATION: u32> PrefixSuffixDecomposition<XLEN>
    for VirtualXORROTWTable<XLEN, ROTATION>
{
    fn suffixes(&self) -> &'static [Suffixes] {
        debug_assert_eq!(XLEN, 64);
        match ROTATION {
            7 => &[Suffixes::One, Suffixes::XorRotW7],
            8 => &[Suffixes::One, Suffixes::XorRotW8],
            12 => &[Suffixes::One, Suffixes::XorRotW12],
            16 => &[Suffixes::One, Suffixes::XorRotW16],
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor_rot] = suffixes.try_into().unwrap();
        match ROTATION {
            7 => prefixes[Prefixes::XorRotW7] * one + xor_rot,
            8 => prefixes[Prefixes::XorRotW8] * one + xor_rot,
            12 => prefixes[Prefixes::XorRotW12] * one + xor_rot,
            16 => prefixes[Prefixes::XorRotW16] * one + xor_rot,
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
    fn mle_random_rotw16() {
        mle_random_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 16>>();
    }

    #[test]
    fn prefix_suffix_rotw16() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 16>>();
    }

    #[test]
    fn mle_random_rotw12() {
        mle_random_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 12>>();
    }

    #[test]
    fn prefix_suffix_rotw12() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 12>>();
    }

    #[test]
    fn mle_random_rotw8() {
        mle_random_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 8>>();
    }

    #[test]
    fn prefix_suffix_rotw8() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 8>>();
    }

    #[test]
    fn mle_random_rotw7() {
        mle_random_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 7>>();
    }

    #[test]
    fn prefix_suffix_rotw7() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTWTable<XLEN, 7>>();
    }

    #[test]
    fn mle_full_hypercube_rotw7() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTWTable<8, 7>>();
    }

    #[test]
    fn mle_full_hypercube_rotw8() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTWTable<8, 8>>();
    }

    #[test]
    fn mle_full_hypercube_rotw12() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTWTable<8, 12>>();
    }

    #[test]
    fn mle_full_hypercube_rotw16() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTWTable<8, 16>>();
    }
}
