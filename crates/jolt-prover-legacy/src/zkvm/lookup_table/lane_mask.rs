use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltLookupTable, PrefixSuffixDecomposition};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

pub(crate) fn lane_value<const WIDTH_BYTES: usize>(address: u64) -> u64 {
    let (base, align_mask) = match WIDTH_BYTES {
        0 => (1, 0),
        1 => (0xff, 0),
        2 => (0xffff, 1),
        4 => (0xffff_ffff, 3),
        _ => unreachable!("unsupported lane width"),
    };
    base << (8 * (address as usize & 7 & !align_mask))
}

fn evaluate_lane_mle<const WIDTH_BYTES: usize, F, C>(r: &[C]) -> F
where
    C: ChallengeFieldOps<F>,
    F: JoltField + FieldChallengeOps<C>,
{
    let (base, first_bit) = match WIDTH_BYTES {
        0 => (1, 0),
        1 => (0xff, 0),
        2 => (0xffff, 1),
        4 => (0xffff_ffff, 2),
        _ => unreachable!("unsupported lane width"),
    };
    let mut result = F::from_u64(base);
    for bit in first_bit..3 {
        let factor = 1u64 << (8 * (1 << bit));
        result *= F::one() + F::from_u64(factor - 1) * r[r.len() - 1 - bit];
    }
    result
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct LaneMaskTable<const XLEN: usize, const WIDTH_BYTES: usize>;

impl<const XLEN: usize, const WIDTH_BYTES: usize> JoltLookupTable
    for LaneMaskTable<XLEN, WIDTH_BYTES>
{
    fn materialize_entry(&self, index: u128) -> u64 {
        lane_value::<WIDTH_BYTES>(index as u64)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        evaluate_lane_mle::<WIDTH_BYTES, F, C>(r)
    }
}

impl<const XLEN: usize, const WIDTH_BYTES: usize> PrefixSuffixDecomposition<XLEN>
    for LaneMaskTable<XLEN, WIDTH_BYTES>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        match WIDTH_BYTES {
            1 => vec![Suffixes::LaneMaskB],
            2 => vec![Suffixes::LaneMaskH],
            4 => vec![Suffixes::LaneMaskW],
            _ => unreachable!("unsupported lane width"),
        }
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix] = suffixes.try_into().unwrap();
        let prefix = match WIDTH_BYTES {
            1 => Prefixes::LaneMaskB,
            2 => Prefixes::LaneMaskH,
            4 => Prefixes::LaneMaskW,
            _ => unreachable!("unsupported lane width"),
        };
        prefixes[prefix] * suffix
    }
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pow2LaneTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for Pow2LaneTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        lane_value::<0>(index as u64)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        evaluate_lane_mle::<0, F, C>(r)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for Pow2LaneTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Pow2Lane]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Pow2Lane] * suffix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use ark_bn254::Fr;
    use common::constants::XLEN;

    macro_rules! table_tests {
        ($module:ident, $table8:ty, $table64:ty) => {
            mod $module {
                use super::*;

                #[test]
                fn mle_full_hypercube() {
                    lookup_table_mle_full_hypercube_test::<Fr, $table8>();
                }

                #[test]
                fn mle_random() {
                    lookup_table_mle_random_test::<Fr, $table64>();
                }

                #[test]
                fn prefix_suffix() {
                    prefix_suffix_test::<XLEN, Fr, $table64>();
                }
            }
        };
    }

    table_tests!(byte, LaneMaskTable<8, 1>, LaneMaskTable<XLEN, 1>);
    table_tests!(halfword, LaneMaskTable<8, 2>, LaneMaskTable<XLEN, 2>);
    table_tests!(word, LaneMaskTable<8, 4>, LaneMaskTable<XLEN, 4>);
    table_tests!(pow2, Pow2LaneTable<8>, Pow2LaneTable<XLEN>);
}
