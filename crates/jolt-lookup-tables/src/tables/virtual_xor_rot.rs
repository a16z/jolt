use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// XOR followed by a fixed right rotation. For LSB-indexed input bits,
/// the MLE is `sum_i 2^((i - R) mod XLEN) * (x_i + y_i - 2*x_i*y_i)`.
/// `evaluate_mle` receives challenges in MSB-first order.
/// At an RV64 decomposition boundary with `2*s` suffix bits, the prefix
/// owns `i >= s` and the suffix owns `i < s`. Rotating preserves disjoint
/// output positions, so `prefix * One + suffix` also holds off the Boolean cube.
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
            61 => &[Prefixes::XorRot61],
            3 => &[Prefixes::XorRot3],
            8 => &[Prefixes::XorRot8],
            9 => &[Prefixes::XorRot9],
            19 => &[Prefixes::XorRot19],
            20 => &[Prefixes::XorRot20],
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
            61 => &[Suffixes::One, Suffixes::XorRot61],
            3 => &[Suffixes::One, Suffixes::XorRot3],
            8 => &[Suffixes::One, Suffixes::XorRot8],
            9 => &[Suffixes::One, Suffixes::XorRot9],
            19 => &[Suffixes::One, Suffixes::XorRot19],
            20 => &[Suffixes::One, Suffixes::XorRot20],
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
            61 => prefixes[Prefixes::XorRot61] * one + xor_rot,
            3 => prefixes[Prefixes::XorRot3] * one + xor_rot,
            8 => prefixes[Prefixes::XorRot8] * one + xor_rot,
            9 => prefixes[Prefixes::XorRot9] * one + xor_rot,
            19 => prefixes[Prefixes::XorRot19] * one + xor_rot,
            20 => prefixes[Prefixes::XorRot20] * one + xor_rot,
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

    #[test]
    fn mle_full_hypercube_rot32() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 32>>();
    }

    #[test]
    fn mle_full_hypercube_rot63() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 63>>();
    }

    #[test]
    fn keccak_rotations_match_mle_and_decomposition() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 2>>();
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 61>>();
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 3>>();
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 8>>();
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 9>>();
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 19>>();
        mle_full_hypercube_test::<8, Fr, VirtualXORROTTable<8, 20>>();
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 2>>();
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 61>>();
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 3>>();
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 8>>();
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 9>>();
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 19>>();
        mle_random_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 20>>();
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 2>>();
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 61>>();
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 3>>();
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 8>>();
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 9>>();
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 19>>();
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTTable<XLEN, 20>>();
    }

    #[test]
    fn keccak_rotations_match_word_reference() {
        use crate::interleave_bits;
        let mut inputs = vec![0, u64::MAX, 0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555];
        for bit in 0..64 {
            inputs.push(1 << bit);
            inputs.push(!(1 << bit));
        }
        for &x in &inputs {
            for &y in &inputs {
                let index = interleave_bits(x, y);
                assert_eq!(
                    VirtualXORROTTable::<64, 2>.materialize_entry(index),
                    (x ^ y).rotate_right(2),
                );
                assert_eq!(
                    VirtualXORROTTable::<64, 61>.materialize_entry(index),
                    (x ^ y).rotate_right(61),
                );
                assert_eq!(
                    VirtualXORROTTable::<64, 3>.materialize_entry(index),
                    (x ^ y).rotate_right(3),
                );
                assert_eq!(
                    VirtualXORROTTable::<64, 8>.materialize_entry(index),
                    (x ^ y).rotate_right(8),
                );
                assert_eq!(
                    VirtualXORROTTable::<64, 9>.materialize_entry(index),
                    (x ^ y).rotate_right(9),
                );
                assert_eq!(
                    VirtualXORROTTable::<64, 19>.materialize_entry(index),
                    (x ^ y).rotate_right(19),
                );
                assert_eq!(
                    VirtualXORROTTable::<64, 20>.materialize_entry(index),
                    (x ^ y).rotate_right(20),
                );
            }
        }
    }
}
