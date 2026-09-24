use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// Halfword store-data shifter:
/// `(x mod 2^(XLEN/4)) << ((XLEN/8)·(y mod 8 & !1))`.
///
/// At XLEN = 64 this is `(rs2 & 0xFFFF) << (8·(ea mod 8 & 6))`. Bit 0 of the
/// offset is ignored (zeroed by the halfword-alignment assert), which keeps
/// the maximum output `0xFFFF << 48` in `u64` range on the full domain.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ShiftDataHTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for ShiftDataHTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let eighth = XLEN / 8;
        let lane = x & (((1u128 << (2 * eighth)) - 1) as u64);
        lane << (eighth as u32 * (y & 6) as u32)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let eighth = XLEN / 8;
        let mut lane = F::zero();
        for k in 0..(2 * eighth) {
            let x_k: F = r[2 * (XLEN - 1 - k)].into();
            lane += x_k * F::from_u64(1u64 << k);
        }
        // offset = 4·y_2 + 2·y_1 (bit 0 ignored)
        let mut result = lane;
        for i in 1..3 {
            let y_i: F = r[2 * (XLEN - 1 - i) + 1].into();
            let scale = F::from_u128((1u128 << (eighth << i)) - 1);
            result = result + result * (scale * y_i);
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ShiftDataHTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::ShiftDataH, Prefixes::OffsetScaleH]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        // The ShiftData/OffsetScale prefix-suffix pairs hardcode the crate
        // XLEN.
        debug_assert_eq!(XLEN, 64);
        &[Suffixes::OffsetScaleH, Suffixes::ShiftDataH]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [offset_scale, shift_data] = suffixes.try_into().unwrap();
        prefixes[Prefixes::ShiftDataH] * offset_scale
            + prefixes[Prefixes::OffsetScaleH] * shift_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{
        mle_full_hypercube_test, mle_random_test, prefix_suffix_materialization_test,
        prefix_suffix_test,
    };
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, ShiftDataHTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ShiftDataHTable<XLEN>>();
    }

    /// Two-round phases put boundaries inside the low six index bits,
    /// exercising every placement of the lane and offset bits relative to
    /// the phase window in the ShiftData/OffsetScale prefix/suffix pairs.
    #[test]
    fn prefix_suffix_small_phases() {
        prefix_suffix_materialization_test::<XLEN, Fr, ShiftDataHTable<XLEN>>(2, 3);
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, ShiftDataHTable<8>>();
    }
}
