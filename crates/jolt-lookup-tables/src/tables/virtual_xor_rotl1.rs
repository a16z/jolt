use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VirtualXORROTL1Table<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for VirtualXORROTL1Table<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (x ^ (y << 1) ^ (y >> (XLEN - 1))) & mask
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
            let y_prev = r[2 * ((i + 1) % XLEN) + 1];
            result += F::from_u64(1 << (XLEN - 1 - i))
                * ((F::one() - x_i) * y_prev + x_i * (F::one() - y_prev));
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualXORROTL1Table<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[
            Prefixes::XorRotL1Acc,
            Prefixes::XorRotL1Straddle,
            Prefixes::XorRotL1Wrap,
        ]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[
            Suffixes::One,
            Suffixes::XorRotL1Pairs,
            Suffixes::TopYBit,
            Suffixes::BottomXBit,
        ]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, pairs, top_y, bottom_x] = suffixes.try_into().unwrap();
        prefixes[Prefixes::XorRotL1Acc] * one
            + pairs
            + prefixes[Prefixes::XorRotL1Straddle] * top_y
            + prefixes[Prefixes::XorRotL1Wrap] * bottom_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, VirtualXORROTL1Table<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, VirtualXORROTL1Table<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROTL1Table<XLEN>>();
    }
}
