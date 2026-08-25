use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::uninterleave_bits;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualNegateIfTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for VirtualNegateIfTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (sign_source, value) = uninterleave_bits(index);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let value = value & mask;
        if sign_source & (1 << (XLEN - 1)) == 0 {
            value
        } else {
            value.wrapping_neg() & mask
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let sign: F = r[0].into();
        let mut value = F::zero();
        let mut value_is_zero = F::one();
        for i in 0..XLEN {
            let bit = r[2 * i + 1];
            value += F::from_u128(1u128 << (XLEN - 1 - i)) * bit;
            value_is_zero *= F::one() - bit;
        }

        let two_to_xlen = F::from_u128(1u128 << XLEN);
        value - F::from_u64(2) * sign * value + two_to_xlen * sign
            - two_to_xlen * sign * value_is_zero
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualNegateIfTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::RightOperand,
            Suffixes::RightOperandIsZero,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_operand, right_operand_is_zero] = suffixes.try_into().unwrap();
        let two_to_xlen = F::from_u128(1u128 << XLEN);

        prefixes[Prefixes::RightOperand] * one + right_operand
            - F::from_u64(2)
                * (prefixes[Prefixes::LeftMsbRightOperand] * one
                    + prefixes[Prefixes::LeftOperandMsb] * right_operand)
            + two_to_xlen * prefixes[Prefixes::LeftOperandMsb] * one
            - two_to_xlen * prefixes[Prefixes::LeftMsbRightOperandIsZero] * right_operand_is_zero
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::VirtualNegateIfTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualNegateIfTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualNegateIfTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualNegateIfTable<XLEN>>();
    }
}
