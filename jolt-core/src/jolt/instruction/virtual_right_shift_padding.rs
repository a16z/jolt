use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::instruction::prefixes::Prefixes;
use crate::jolt::subtable::LassoSubtable;
use crate::poly::eq_poly::EqPolynomial;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::index_to_field_bitvector;
use crate::utils::math::Math;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct RightShiftPaddingInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for RightShiftPaddingInstruction<WORD_SIZE> {
    fn to_lookup_index(&self) -> u64 {
        self.0
    }

    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, _: &[F], _: usize, _: usize) -> F {
        F::zero()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        _: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![]
    }

    fn to_indices(&self, C: usize, _: usize) -> Vec<usize> {
        vec![0; C]
    }

    fn lookup_entry(&self) -> u64 {
        let shift = self.0 % WORD_SIZE as u64;
        let ones = (1 << shift) - 1;
        ones << (WORD_SIZE as u64 - shift)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let shift = index % WORD_SIZE as u64;
        let ones = (1 << shift) - 1;
        ones << (WORD_SIZE as u64 - shift)
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
            let padding = ((1 << shift) - 1) << (WORD_SIZE - shift);
            result += F::from_u64(padding)
                * eq.evaluate(&index_to_field_bitvector(shift as u64, WORD_SIZE.log_2()))
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for RightShiftPaddingInstruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::RightShiftPadding]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_shift_padding] = suffixes.try_into().unwrap();
        // 2^WORD_SIZE - 2^shift = 0b11...100..0
        F::from_u64(1 << WORD_SIZE) * one
            - prefixes[Prefixes::RightShiftPadding] * right_shift_padding
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::RightShiftPaddingInstruction;
    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    #[test]
    fn right_shift_padding_materialize_entry() {
        materialize_entry_test::<Fr, RightShiftPaddingInstruction<32>>();
    }

    #[test]
    fn right_shift_padding_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, RightShiftPaddingInstruction<8>>();
    }

    #[test]
    fn right_shift_padding_mle_random() {
        instruction_mle_random_test::<Fr, RightShiftPaddingInstruction<32>>();
    }

    #[test]
    fn right_shift_padding_prefix_suffix() {
        prefix_suffix_test::<Fr, RightShiftPaddingInstruction<32>>();
    }
}
