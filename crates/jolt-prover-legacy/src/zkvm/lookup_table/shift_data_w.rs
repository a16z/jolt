use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::uninterleave_bits;

/// Word store-data shifter:
/// `(x mod 2^(XLEN/2)) << ((XLEN/8)·(y mod 8 & 4))`.
///
/// At XLEN = 64 this is `(rs2 & 0xFFFFFFFF) << (8·(ea mod 8 & 4))`. Only bit
/// 2 of the offset is read (bits 0-1 are zeroed by the word-alignment
/// assert), keeping the maximum output `0xFFFFFFFF << 32` in `u64` range on
/// the full domain.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShiftDataWTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for ShiftDataWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let eighth = XLEN / 8;
        let lane = x & (((1u128 << (4 * eighth)) - 1) as u64);
        lane << (eighth as u32 * (y & 4) as u32)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let eighth = XLEN / 8;
        let mut lane = F::zero();
        for k in 0..(4 * eighth) {
            let x_k: F = r[2 * (XLEN - 1 - k)].into();
            lane += x_k * F::from_u64(1u64 << k);
        }
        // offset = 4·y_2 (bits 0-1 ignored)
        let y_2: F = r[2 * (XLEN - 1 - 2) + 1].into();
        let scale = F::from_u128((1u128 << (eighth << 2)) - 1);
        lane + lane * (scale * y_2)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ShiftDataWTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::OffsetScaleW, Suffixes::ShiftDataW]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [offset_scale, shift_data] = suffixes.try_into().unwrap();
        prefixes[Prefixes::ShiftDataW] * offset_scale
            + prefixes[Prefixes::OffsetScaleW] * shift_data
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::ShiftDataWTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        prefix_suffix_test_with_phase_size,
    };
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ShiftDataWTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ShiftDataWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ShiftDataWTable<XLEN>>();
    }

    /// Two-round phases put boundaries inside the low six index bits,
    /// exercising every placement of the lane and offset bits relative to
    /// the phase window in the ShiftData/OffsetScale prefix/suffix pairs.
    #[test]
    fn prefix_suffix_small_phases() {
        prefix_suffix_test_with_phase_size::<XLEN, Fr, ShiftDataWTable<XLEN>>(2, 20);
    }
}
