use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// `v` rotated right by `rotation` bits within an `XLEN`-bit word.
pub(crate) fn rotate_right_xlen<const XLEN: usize>(v: u64, rotation: u32) -> u64 {
    let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
    let r = rotation as usize % XLEN;
    let v = (v & mask) as u128;
    (((v >> r) | (v << (XLEN - r))) as u64) & mask
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualXORROTTable<const XLEN: usize, const ROTATION: u32>;

impl<const XLEN: usize, const ROTATION: u32> VirtualXORROTTable<XLEN, ROTATION> {
    const PREFIX: Prefixes = Prefixes::xor_rot(ROTATION);
    const PREFIXES: &'static [Prefixes] = &[Self::PREFIX];
    const SUFFIXES: &'static [Suffixes] = &[Suffixes::One, Suffixes::xor_rot(ROTATION)];
}

impl<const XLEN: usize, const ROTATION: u32> LookupTable for VirtualXORROTTable<XLEN, ROTATION> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        rotate_right_xlen::<XLEN>(x ^ y, ROTATION)
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
        Self::PREFIXES
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        debug_assert_eq!(XLEN, 64);
        Self::SUFFIXES
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor_rot] = suffixes.try_into().unwrap();
        prefixes[Self::PREFIX] * one + xor_rot
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
