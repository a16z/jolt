use crate::{
    field::JoltField,
    jolt_onnx::subtable::{higher::HigherHalfSubtable, lower::LowerHalfSubtable},
    subprotocols::sparse_dense_shout::PrefixSuffixDecomposition,
    utils::{instruction_utils::concatenate_lookups, uninterleave_bits},
};
use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::jolt::instruction::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltInstruction,
};
use crate::{
    jolt::{
        instruction::SubtableIndices,
        subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MinInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for MinInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);
        let lt = {
            let ltu = vals_by_subtable[0];
            let eq = vals_by_subtable[1];

            let mut sum = F::zero();
            let mut eq_prod = F::one();

            for i in 0..C - 1 {
                sum += ltu[i] * eq_prod;
                eq_prod *= eq[i];
            }
            // Do not need to update `eq_prod` for the last iteration
            sum + ltu[C - 1] * eq_prod
        };
        let x = concatenate_lookups(vals_by_subtable[2], C, log2(M) as usize / 2);
        let y = concatenate_lookups(vals_by_subtable[3], C, log2(M) as usize / 2);
        (F::one() - lt) * y + lt * x
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C + 2
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C - 1)),
            (
                Box::new(HigherHalfSubtable::new()),
                SubtableIndices::from(0..C),
            ),
            (
                Box::new(LowerHalfSubtable::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn materialize_entry(&self, _: u64) -> u64 {
        unimplemented!("for shout we will use a simpler min instruction implementation.")
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for 32-bit and 64-bit word sizes
        (self.0).min(self.1)
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

    fn evaluate_mle<F: JoltField>(&self, _: &[F]) -> F {
        unimplemented!("for shout we will use a simpler min instruction implementation.")
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for MinInstruction<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LessThan] * one + prefixes[Prefixes::Eq] * less_than
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

    use super::MinInstruction;

    #[test]
    fn sltu_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = MinInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            jolt_instruction_test!(MinInstruction::<WORD_SIZE>(x, x));
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            MinInstruction::<WORD_SIZE>(100, 0),
            MinInstruction::<WORD_SIZE>(0, 100),
            MinInstruction::<WORD_SIZE>(1, 0),
            MinInstruction::<WORD_SIZE>(0, u32_max),
            MinInstruction::<WORD_SIZE>(u32_max, 0),
            MinInstruction::<WORD_SIZE>(u32_max, u32_max),
            MinInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            MinInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn sltu_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = MinInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u64();
            jolt_instruction_test!(MinInstruction::<WORD_SIZE>(x, x));
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            MinInstruction::<WORD_SIZE>(100, 0),
            MinInstruction::<WORD_SIZE>(0, 100),
            MinInstruction::<WORD_SIZE>(1, 0),
            MinInstruction::<WORD_SIZE>(0, u64_max),
            MinInstruction::<WORD_SIZE>(u64_max, 0),
            MinInstruction::<WORD_SIZE>(u64_max, u64_max),
            MinInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            MinInstruction::<WORD_SIZE>(1 << 32, u64_max),
            MinInstruction::<WORD_SIZE>(1 << 63, 1 << (63 - 1)),
            MinInstruction::<WORD_SIZE>(1 << (63 - 1), 1 << 63),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
