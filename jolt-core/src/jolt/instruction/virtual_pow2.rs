use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::LassoSubtable;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::math::Math;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct POW2Instruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for POW2Instruction<WORD_SIZE> {
    fn to_lookup_index(&self) -> u64 {
        self.0
    }

    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        todo!()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        todo!()
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        todo!()
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        todo!()
    }

    fn lookup_entry(&self) -> u64 {
        1 << (self.0 % WORD_SIZE as u64)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        1 << (index % WORD_SIZE as u64)
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
        let mut result = F::one();
        for i in 0..WORD_SIZE.log_2() {
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }
}

impl<const WORD_SIZE: usize, F: JoltField> PrefixSuffixDecomposition<WORD_SIZE, F>
    for POW2Instruction<WORD_SIZE>
{
    fn prefixes() -> Vec<Prefixes> {
        vec![]
    }

    fn suffixes() -> Vec<Suffixes> {
        vec![Suffixes::Pow2]
    }

    fn combine(_: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        suffixes[Suffixes::Pow2]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::POW2Instruction;
    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    #[test]
    fn pow2_materialize_entry() {
        materialize_entry_test::<Fr, POW2Instruction<32>>();
    }

    #[test]
    fn pow2_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, POW2Instruction<8>>();
    }

    #[test]
    fn pow2_mle_random() {
        instruction_mle_random_test::<Fr, POW2Instruction<32>>();
    }

    #[test]
    fn pow2_prefix_suffix() {
        prefix_suffix_test::<Fr, POW2Instruction<32>>();
    }
}
