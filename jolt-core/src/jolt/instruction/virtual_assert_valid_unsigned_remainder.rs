use crate::{
    field::JoltField, jolt::subtable::right_is_zero::RightIsZeroSubtable, utils::uninterleave_bits,
};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{
    jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssertValidUnsignedRemainderInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction
    for AssertValidUnsignedRemainderInstruction<WORD_SIZE>
{
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);
        let ltu = vals_by_subtable[0];
        let eq = vals_by_subtable[1];
        let divisor_is_zero: F = vals_by_subtable[2].iter().product();

        let mut sum = F::zero();
        let mut eq_prod = F::one();

        for i in 0..C - 1 {
            sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        // LTU(x, y) + EQ(y, 0)
        sum + ltu[C - 1] * eq_prod + divisor_is_zero
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
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C - 1)),
            (
                Box::new(RightIsZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Same for both 32-bit and 64-bit word sizes
        let remainder = self.0;
        let divisor = self.1;
        (divisor == 0 || remainder < divisor).into()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (remainder, divisor) = uninterleave_bits(index);
        (divisor == 0 || remainder < divisor).into()
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
        let mut divisor_is_zero = F::one();
        let mut lt = F::zero();
        let mut eq = F::one();

        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            divisor_is_zero *= F::one() - y_i;
            lt += (F::one() - x_i) * y_i * eq;
            eq *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }

        lt + divisor_is_zero
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for AssertValidUnsignedRemainderInstruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LessThan,
            Suffixes::RightOperandIsZero,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than, right_operand_is_zero] = suffixes.try_into().unwrap();
        // divisor == 0 || remainder < divisor
        prefixes[Prefixes::RightOperandIsZero] * right_operand_is_zero
            + prefixes[Prefixes::LessThan] * one
            + prefixes[Prefixes::Eq] * less_than
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

    use super::AssertValidUnsignedRemainderInstruction;

    #[test]
    fn assert_valid_unsigned_remainder_materialize_entry() {
        materialize_entry_test::<Fr, AssertValidUnsignedRemainderInstruction<32>>();
    }

    #[test]
    fn assert_valid_unsigned_remainder_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, AssertValidUnsignedRemainderInstruction<8>>();
    }

    #[test]
    fn assert_valid_unsigned_remainder_mle_random() {
        instruction_mle_random_test::<Fr, AssertValidUnsignedRemainderInstruction<32>>();
    }

    #[test]
    fn assert_valid_unsigned_remainder_prefix_suffix() {
        prefix_suffix_test::<Fr, AssertValidUnsignedRemainderInstruction<32>>();
    }

    #[test]
    fn assert_valid_unsigned_remainder_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            jolt_instruction_test!(AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(x, x));
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(100, 0),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(0, 100),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(1, 0),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(0, u32_max),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u32_max, 0),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u32_max, u32_max),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn assert_valid_unsigned_remainder_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(x, x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(100, 0),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(0, 100),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(1, 0),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(0, u64_max),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u64_max, 0),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u64_max, u64_max),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(1 << 8, u64_max),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u64_max, u64_max - 1),
            AssertValidUnsignedRemainderInstruction::<WORD_SIZE>(u64_max - 1, u64_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
