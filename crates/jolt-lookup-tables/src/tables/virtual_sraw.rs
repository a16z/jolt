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
pub struct VirtualSRAWTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for VirtualSRAWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let mut x = LookupBits::new(x as u128, XLEN);
        let mut y = LookupBits::new(y as u128, XLEN);
        let half = XLEN / 2;

        for _ in 0..half {
            let _ = x.pop_msb();
            let _ = y.pop_msb();
        }
        let sign_bit = u64::from(x.leading_ones() != 0);
        let mut entry = 0u64;
        let mut sign_extension = ((1u128 << XLEN) - (1u128 << half)) as u64;
        for i in 0..half {
            let x_i = x.pop_msb() as u64;
            let y_i = y.pop_msb() as u64;
            entry = entry * (1 + y_i) + x_i * y_i;
            if i != 0 {
                sign_extension += (1 << i) * (1 - y_i);
            }
        }
        entry + sign_bit * sign_extension
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let half = XLEN / 2;
        let mut result = F::zero();
        let mut sign_extension = F::from_u128((1u128 << XLEN) - (1u128 << half));
        for i in 0..half {
            let x_i = r[XLEN + 2 * i];
            let y_i = r[XLEN + 2 * i + 1];
            result *= F::one() + y_i;
            result += x_i * y_i;
            if i != 0 {
                sign_extension += F::from_u128(1u128 << i) * (F::one() - y_i);
            }
        }
        result + r[XLEN] * sign_extension
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualSRAWTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[
            Prefixes::RightShiftW,
            Prefixes::WordMsb,
            Prefixes::SignExtensionW,
        ]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::RightShiftW,
            Suffixes::RightShiftWHelper,
            Suffixes::SignExtensionW,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, right_shift_w, right_shift_w_helper, sign_extension_w] =
            suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShiftW] * right_shift_w_helper
            + right_shift_w
            + prefixes[Prefixes::WordMsb] * sign_extension_w
            + prefixes[Prefixes::SignExtensionW] * one
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
        mle_random_test::<XLEN, Fr, VirtualSRAWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualSRAWTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, VirtualSRAWTable<8>>();
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
                let expected = ((x as u32 as i32) >> shift) as i64 as u64;
                assert_eq!(
                    VirtualSRAWTable::<64>.materialize_entry(interleave_bits(x, y)),
                    expected,
                    "x={x:#x}, shift={shift}",
                );
            }
        }
    }
}
