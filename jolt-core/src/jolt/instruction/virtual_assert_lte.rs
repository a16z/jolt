use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{
    field::JoltField,
    jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    utils::{instruction_utils::chunk_and_concatenate_operands, uninterleave_bits},
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ASSERTLTEInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for ASSERTLTEInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);
        let ltu = vals_by_subtable[0];
        let eq = vals_by_subtable[1];

        // Accumulator for LTU(x, y)
        let mut ltu_sum = F::zero();
        // Accumulator for EQ(x, y)
        let mut eq_prod = F::one();

        for i in 0..C {
            ltu_sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }

        // LTU(x,y) || EQ(x,y)
        ltu_sum + eq_prod
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Same for both 32-bit and 64-bit word sizes
        (self.0 <= self.1).into()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x <= y).into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8), rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64, rng.next_u32() as u64),
            64 => Self(rng.next_u64(), rng.next_u64()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut lt = F::zero();
        let mut eq = F::one();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            lt += (F::one() - x_i) * y_i * eq;
            eq *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }

        lt + eq
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for ASSERTLTEInstruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan, Suffixes::Eq]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than, eq] = suffixes.try_into().unwrap();
        // LT(x, y) + EQ(x, y)
        prefixes[Prefixes::LessThan] * one
            + prefixes[Prefixes::Eq] * less_than
            + prefixes[Prefixes::Eq] * eq
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{
        jolt::instruction::{
            test::{
                instruction_mle_full_hypercube_test, instruction_mle_random_test,
                materialize_entry_test, prefix_suffix_test,
            },
            JoltInstruction,
        },
        jolt_instruction_test,
    };

    use super::ASSERTLTEInstruction;

    #[test]
    fn assert_lte_materialize_entry() {
        materialize_entry_test::<Fr, ASSERTLTEInstruction<32>>();
    }

    #[test]
    fn assert_lte_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, ASSERTLTEInstruction<8>>();
    }

    #[test]
    fn assert_lte_mle_random() {
        instruction_mle_random_test::<Fr, ASSERTLTEInstruction<32>>();
    }

    #[test]
    fn assert_lte_prefix_suffix() {
        prefix_suffix_test::<Fr, ASSERTLTEInstruction<32>>();
    }

    #[test]
    fn assert_lte_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32();
            let y = rng.next_u32();

            let instruction = ASSERTLTEInstruction::<WORD_SIZE>(x as u64, y as u64);

            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u32();
            jolt_instruction_test!(ASSERTLTEInstruction::<WORD_SIZE>(x as u64, x as u64));
        }

        // Edge-cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ASSERTLTEInstruction::<WORD_SIZE>(100, 0),
            ASSERTLTEInstruction::<WORD_SIZE>(0, 100),
            ASSERTLTEInstruction::<WORD_SIZE>(1, 0),
            ASSERTLTEInstruction::<WORD_SIZE>(0, u32_max),
            ASSERTLTEInstruction::<WORD_SIZE>(u32_max, 0),
            ASSERTLTEInstruction::<WORD_SIZE>(u32_max, u32_max),
            ASSERTLTEInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            ASSERTLTEInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn assert_lte_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let y = rng.next_u64();

            let instruction = ASSERTLTEInstruction::<WORD_SIZE>(x, y);

            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u64();
            jolt_instruction_test!(ASSERTLTEInstruction::<WORD_SIZE>(x, x));
        }

        // Edge-cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            ASSERTLTEInstruction::<WORD_SIZE>(100, 0),
            ASSERTLTEInstruction::<WORD_SIZE>(0, 100),
            ASSERTLTEInstruction::<WORD_SIZE>(1, 0),
            ASSERTLTEInstruction::<WORD_SIZE>(0, u64_max),
            ASSERTLTEInstruction::<WORD_SIZE>(u64_max, 0),
            ASSERTLTEInstruction::<WORD_SIZE>(u64_max, u64_max),
            ASSERTLTEInstruction::<WORD_SIZE>(u64_max, u64_max - 1),
            ASSERTLTEInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            ASSERTLTEInstruction::<WORD_SIZE>(1 << 8, u64_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
