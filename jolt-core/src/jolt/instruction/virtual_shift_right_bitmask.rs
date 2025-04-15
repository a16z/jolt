use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;
use crate::field::JoltField;
use crate::jolt::instruction::prefixes::Prefixes;
use crate::poly::eq_poly::EqPolynomial;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::index_to_field_bitvector;
use crate::utils::math::Math;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShiftRightBitmaskInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for ShiftRightBitmaskInstruction<WORD_SIZE> {
    fn to_lookup_index(&self) -> u64 {
        self.0
    }

    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn lookup_entry(&self) -> u64 {
        let shift = self.0 % WORD_SIZE as u64;
        let ones = (1 << (WORD_SIZE - shift as usize)) - 1;
        ones << shift
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let shift = index % WORD_SIZE as u64;
        let ones = (1 << (WORD_SIZE - shift as usize)) - 1;
        ones << shift
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64),
            64 => Self(rng.next_u64()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let eq = EqPolynomial::new(r[r.len() - WORD_SIZE.log_2()..].to_vec());
        let mut result = F::zero();
        for shift in 0..WORD_SIZE {
            let bitmask = ((1 << (WORD_SIZE - shift)) - 1) << shift;
            result += F::from_u64(bitmask)
                * eq.evaluate(&index_to_field_bitvector(shift as u64, WORD_SIZE.log_2()))
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for ShiftRightBitmaskInstruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Pow2]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, pow2] = suffixes.try_into().unwrap();
        // 2^WORD_SIZE - 2^shift = 0b11...100..0
        F::from_u64(1 << WORD_SIZE) * one - prefixes[Prefixes::Pow2] * pow2
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::ShiftRightBitmaskInstruction;
    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    #[test]
    fn right_shift_bitmask_materialize_entry() {
        materialize_entry_test::<Fr, ShiftRightBitmaskInstruction<32>>();
    }

    #[test]
    fn right_shift_bitmask_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, ShiftRightBitmaskInstruction<8>>();
    }

    #[test]
    fn right_shift_bitmask_mle_random() {
        instruction_mle_random_test::<Fr, ShiftRightBitmaskInstruction<32>>();
    }

    #[test]
    fn right_shift_bitmask_prefix_suffix() {
        prefix_suffix_test::<Fr, ShiftRightBitmaskInstruction<32>>();
    }
}
