use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualSRATable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for VirtualSRATable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let mut x = LookupBits::new(x as u128, XLEN);
        let mut y = LookupBits::new(y as u128, XLEN);

        let sign_bit = u64::from(x.leading_ones() != 0);
        let mut entry = 0;
        let mut sign_extension = 0;
        for i in 0..XLEN {
            let x_i = x.pop_msb() as u64;
            let y_i = y.pop_msb() as u64;
            entry *= 1 + y_i;
            entry += x_i * y_i;
            if i != 0 {
                sign_extension += (1 << i) * (1 - y_i);
            }
        }
        entry + sign_bit * sign_extension
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        let mut sign_extension = F::zero();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result *= F::one() + y_i;
            result += x_i * y_i;
            if i != 0 {
                sign_extension += F::from_u64(1 << i) * (F::one() - y_i);
            }
        }
        result + r[0] * sign_extension
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualSRATable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::RightShift,
            Suffixes::RightShiftHelper,
            Suffixes::SignExtension,
        ]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_shift, right_shift_helper, sign_extension] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * right_shift_helper
            + right_shift
            + prefixes[Prefixes::LeftOperandMsb] * sign_extension
            + prefixes[Prefixes::SignExtension] * one
    }

    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        crate::tables::test_utils::gen_bitmask_lookup_index::<XLEN>(rng)
    }
}
