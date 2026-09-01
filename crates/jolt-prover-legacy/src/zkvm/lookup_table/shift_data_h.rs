use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::uninterleave_bits;

/// Halfword store-data shifter:
/// `(x mod 2^(XLEN/4)) << ((XLEN/8)·(y mod 8 & !1))`.
///
/// At XLEN = 64 this is `(rs2 & 0xFFFF) << (8·(ea mod 8 & 6))`. Bit 0 of the
/// offset is ignored (zeroed by the halfword-alignment assert), which keeps
/// the maximum output `0xFFFF << 48` in `u64` range on the full domain.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShiftDataHTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for ShiftDataHTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let eighth = XLEN / 8;
        let lane = x & (((1u128 << (2 * eighth)) - 1) as u64);
        lane << (eighth as u32 * (y & 6) as u32)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
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
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::OffsetScaleH, Suffixes::ShiftDataH]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [offset_scale, shift_data] = suffixes.try_into().unwrap();
        prefixes[Prefixes::ShiftDataH] * offset_scale
            + prefixes[Prefixes::OffsetScaleH] * shift_data
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::ShiftDataHTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        prefix_suffix_test_with_phase_size,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ShiftDataHTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ShiftDataHTable<XLEN>>();
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
        prefix_suffix_test_with_phase_size::<XLEN, Fr, ShiftDataHTable<XLEN>>(2, 20);
    }
}
