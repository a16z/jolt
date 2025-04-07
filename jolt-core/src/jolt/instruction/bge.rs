use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{slt::SLTInstruction, JoltInstruction, SubtableIndices};
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{
    field::JoltField,
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable,
        ltu::LtuSubtable, right_msb::RightMSBSubtable, LassoSubtable,
    },
    utils::{instruction_utils::chunk_and_concatenate_operands, uninterleave_bits},
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct BGEInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for BGEInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // 1 - SLT(x, y) =
        F::one() - SLTInstruction::<WORD_SIZE>(self.0, self.1).combine_lookups(vals, C, M)
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C + 1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LeftMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(RightMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(1..C - 1)),
            (Box::new(LtAbsSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqAbsSubtable::new()), SubtableIndices::from(0)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        if WORD_SIZE == 32 {
            let x = self.0 as i32;
            let y = self.1 as i32;
            (x >= y) as u64
        } else if WORD_SIZE == 64 {
            let x = self.0 as i64;
            let y = self.1 as i64;
            (x >= y) as u64
        } else {
            panic!("BGE is only implemented for 32-bit or 64-bit word sizes")
        }
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as i8 >= y as i8).into(),
            32 => (x as i32 >= y as i32).into(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
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
        F::one() - SLTInstruction::<WORD_SIZE>::default().evaluate_mle(r)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for BGEInstruction<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        // 1 - LT(x, y) = 1 - (isNegative(x) && isPositive(y)) - LTU(x, y)
        one + prefixes[Prefixes::RightOperandMsb] * one
            - prefixes[Prefixes::LeftOperandMsb] * one
            - prefixes[Prefixes::LessThan] * one
            - prefixes[Prefixes::Eq] * less_than
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

    use super::BGEInstruction;

    #[test]
    fn bge_materialize_entry() {
        materialize_entry_test::<Fr, BGEInstruction<32>>();
    }

    #[test]
    fn bge_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, BGEInstruction<8>>();
    }

    #[test]
    fn bge_mle_random() {
        instruction_mle_random_test::<Fr, BGEInstruction<32>>();
    }

    #[test]
    fn bge_prefix_suffix() {
        prefix_suffix_test::<Fr, BGEInstruction<32>>();
    }

    #[test]
    fn bge_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32();
            let y = rng.next_u32();

            let instruction = BGEInstruction::<WORD_SIZE>(x as u64, y as u64);

            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u32();
            jolt_instruction_test!(BGEInstruction::<WORD_SIZE>(x as u64, x as u64));
        }

        // Edge-cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            BGEInstruction::<WORD_SIZE>(100, 0),
            BGEInstruction::<WORD_SIZE>(0, 100),
            BGEInstruction::<WORD_SIZE>(1, 0),
            BGEInstruction::<WORD_SIZE>(0, u32_max),
            BGEInstruction::<WORD_SIZE>(u32_max, 0),
            BGEInstruction::<WORD_SIZE>(u32_max, u32_max),
            BGEInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            BGEInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn bge_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let instruction = BGEInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u64();
            jolt_instruction_test!(BGEInstruction::<WORD_SIZE>(x, x));
        }

        // Edge-cases
        let i64_min = i64::MIN as u64;
        let i64_max = i64::MAX as u64;
        let instructions = vec![
            BGEInstruction::<WORD_SIZE>(100, 0),
            BGEInstruction::<WORD_SIZE>(0, 100),
            BGEInstruction::<WORD_SIZE>(1, 1),
            BGEInstruction::<WORD_SIZE>(0, i64_max),
            BGEInstruction::<WORD_SIZE>(i64_max, 0),
            BGEInstruction::<WORD_SIZE>(i64_max, i64_max),
            BGEInstruction::<WORD_SIZE>(i64_max, 1 << 32),
            BGEInstruction::<WORD_SIZE>(1 << 32, i64_max),
            BGEInstruction::<WORD_SIZE>(i64_min, 0),
            BGEInstruction::<WORD_SIZE>(0, i64_min),
            BGEInstruction::<WORD_SIZE>(i64_min, i64_max),
            BGEInstruction::<WORD_SIZE>(i64_max, i64_min),
            BGEInstruction::<WORD_SIZE>(-1i64 as u64, 0),
            BGEInstruction::<WORD_SIZE>(0, -1i64 as u64),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
