use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltLookupTable, PrefixSuffixDecomposition};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::{lookup_bits::LookupBits, uninterleave_bits};

pub(crate) fn signed_extract<const XLEN: usize>(index: u128) -> u64 {
    let (x, y) = uninterleave_bits(index);
    let mut x = LookupBits::new(x as u128, XLEN);
    let mut y = LookupBits::new(y as u128, XLEN);
    let mut packed = 0u128;
    let mut top_count = 0u128;
    let mut signed_weight = 0u128;
    let mut previous_mask = 0u128;

    while x.len() != 0 {
        let x_i = u128::from(x.pop_msb());
        let y_i = u128::from(y.pop_msb());
        packed = packed * (1 + y_i) + x_i * y_i;
        let top = x_i * y_i * (1 - previous_mask);
        top_count += top;
        signed_weight = signed_weight * (1 + y_i) + top;
        previous_mask = y_i;
    }

    (packed + (1u128 << XLEN) * top_count - 2 * signed_weight) as u64
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct LaneExtractSTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for LaneExtractSTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        signed_extract::<XLEN>(index)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut packed = F::zero();
        let mut top_count = F::zero();
        let mut signed_weight = F::zero();
        let mut previous_mask = F::zero();

        for pair in r.chunks_exact(2) {
            let x_i = pair[0];
            let y_i = pair[1];
            packed = packed * (F::one() + y_i) + x_i * y_i;
            let top = x_i * y_i * (F::one() - previous_mask);
            top_count += top;
            signed_weight = signed_weight * (F::one() + y_i) + top;
            previous_mask = y_i.into();
        }

        packed + F::from_u128(1u128 << XLEN) * top_count - F::from_u64(2) * signed_weight
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for LaneExtractSTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::RightShiftHelper,
            Suffixes::LaneExtractValue,
            Suffixes::LaneExtractStraddle,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, helper, value, straddle] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * helper
            + prefixes[Prefixes::LaneSignS] * helper
            + prefixes[Prefixes::LaneSignT] * one
            + value
            - prefixes[Prefixes::LastMaskBit] * straddle
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        use rand::Rng;

        let x = rng.gen::<u64>() & ((1u128 << XLEN).wrapping_sub(1) as u64);
        let start = rng.gen_range(0..XLEN);
        let width = rng.gen_range(1..=XLEN - start);
        let y = ((((1u128 << width) - 1) << start) & ((1u128 << XLEN) - 1)) as u64;
        crate::utils::interleave_bits(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_valid_index_test,
        prefix_suffix_test,
    };
    use ark_bn254::Fr;
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, LaneExtractSTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_valid_index_test::<XLEN, Fr, LaneExtractSTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, LaneExtractSTable<XLEN>>();
    }
}
