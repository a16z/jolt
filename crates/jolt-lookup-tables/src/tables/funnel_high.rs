use jolt_field::JoltField;
#[cfg(test)]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// High half of a funnel shift selected by a leading-ones mask: for
/// `y = !0 << s` this is `(x mod 2^s) << (XLEN − s)`, and 0 when `s = 0`.
///
/// On every input, each bit of `x` at a zero of `y` moves up by the number of
/// ones of `y` above it (the left-shift half of `VirtualROTR`). Those targets
/// are distinct and below `XLEN`, so the output always fits in `XLEN` bits.
/// The prefix/suffix decomposition holds on leading-ones masks only, which is
/// all a `WindowMaskD` lookup produces.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FunnelHighTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for FunnelHighTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x_bits, y_bits) = uninterleave_bits(index);
        let mut ones_above = 0;
        let mut result = 0u64;
        for i in (0..XLEN).rev() {
            if (y_bits >> i) & 1 == 1 {
                ones_above += 1;
            } else {
                result |= ((x_bits >> i) & 1) << (i + ones_above);
            }
        }
        result
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        assert_eq!(r.len(), 2 * XLEN, "r must have length 2 * XLEN");

        let mut prod_one_plus_y = F::one();
        let mut result = F::zero();
        for (i, chunk) in r.chunks_exact(2).enumerate() {
            let r_x = chunk[0];
            let r_y = chunk[1];
            result += r_x * (F::one() - r_y) * prod_one_plus_y * F::from_u64(1 << (XLEN - 1 - i));
            prod_one_plus_y *= F::one() + r_y;
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for FunnelHighTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::LeftShiftHelper, Prefixes::LeftShift]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::LeftShift, Suffixes::One]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [left_shift, one] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LeftShiftHelper] * left_shift + prefixes[Prefixes::LeftShift] * one
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut StdRng) -> u128 {
        crate::tables::test_utils::gen_bitmask_lookup_index::<XLEN>(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleave_bits;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, FunnelHighTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, FunnelHighTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, FunnelHighTable<8>>();
    }

    #[test]
    fn byte_masks_move_the_low_bytes_to_the_top() {
        let x = 0x0123_4567_89AB_CDEFu64;
        for offset in 0..8 {
            let mask = u64::MAX << (8 * offset);
            let expected = if offset == 0 {
                0
            } else {
                x << (64 - 8 * offset)
            };
            assert_eq!(
                FunnelHighTable::<XLEN>.materialize_entry(interleave_bits(x, mask)),
                expected,
                "offset {offset}"
            );
        }
    }
}
