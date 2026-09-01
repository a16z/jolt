use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualSRLWTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for VirtualSRLWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let mut x = LookupBits::new(x as u128, XLEN);
        let mut y = LookupBits::new(y as u128, XLEN);
        let half = XLEN / 2;

        for _ in 0..half {
            let _ = x.pop_msb();
            let _ = y.pop_msb();
        }
        let sign_bit = x.leading_ones() != 0;
        let mut entry = 0u64;
        let mut y_0 = 0;
        for _ in 0..half {
            let x_i = x.pop_msb() as u64;
            let y_i = y.pop_msb() as u64;
            entry = entry * (1 + y_i) + x_i * y_i;
            y_0 = y_i;
        }
        let extension = ((1u128 << XLEN) - (1u128 << half)) as u64;
        entry + u64::from(sign_bit) * y_0 * extension
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let half = XLEN / 2;
        let mut result = F::zero();
        for i in half..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result *= F::one() + y_i;
            result += x_i * y_i;
        }
        result + r[XLEN] * r[2 * XLEN - 1] * F::from_u128((1u128 << XLEN) - (1u128 << half))
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualSRLWTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::RightShiftW, Prefixes::SrlwSext]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::RightShiftW,
            Suffixes::RightShiftWHelper,
            Suffixes::Lsb,
            Suffixes::X31Y0,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [right_shift_w, right_shift_w_helper, lsb, x31_y0] = suffixes.try_into().unwrap();
        let extension = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2)));
        prefixes[Prefixes::RightShiftW] * right_shift_w_helper
            + right_shift_w
            + extension * (prefixes[Prefixes::SrlwSext] * lsb + x31_y0)
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        crate::tables::test_utils::gen_bitmask_w_lookup_index::<XLEN>(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleave::interleave_bits;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, VirtualSRLWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualSRLWTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, VirtualSRLWTable<8>>();
    }

    #[test]
    fn materialize_matches_word_shift() {
        let values = [
            0,
            1,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0xa5a5_a5a5_8000_0001,
        ];
        for x in values {
            for shift in 0..32 {
                let y = (1u64 << 32) - (1u64 << shift);
                let expected = ((x as u32) >> shift) as i32 as i64 as u64;
                assert_eq!(
                    VirtualSRLWTable::<64>.materialize_entry(interleave_bits(x, y)),
                    expected,
                    "x={x:#x}, shift={shift}",
                );
            }
        }
    }
}
