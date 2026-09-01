use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltLookupTable, PrefixSuffixDecomposition};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::lookup_bits::LookupBits;
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualSRAWTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for VirtualSRAWTable<XLEN> {
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
        let mut entry = 0;
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
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
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
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::RightShiftW,
            Suffixes::RightShiftWHelper,
            Suffixes::SignExtensionW,
        ]
    }

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
        super::test::gen_bitmask_w_lookup_index(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::interleave_bits;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use ark_bn254::Fr;
    use common::constants::XLEN;

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualSRAWTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualSRAWTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualSRAWTable<8>>();
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
