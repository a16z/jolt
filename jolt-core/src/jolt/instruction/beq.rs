use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::JoltInstruction;
use crate::{
    field::JoltField,
    jolt::{
        instruction::SubtableIndices,
        subtable::{eq::EqSubtable, LassoSubtable},
    },
    subprotocols::sparse_dense_shout::{current_suffix_len, LookupBits, SparseDenseSumcheckAlt},
    utils::{
        instruction_utils::chunk_and_concatenate_operands, interleave_bits, uninterleave_bits,
    },
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct BEQInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for BEQInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, _: usize) -> F {
        vals.iter().product::<F>()
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(EqSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert!(r.len() % 2 == 0);

        let x = r.iter().step_by(2);
        let y = r.iter().skip(1).step_by(2);
        x.zip(y)
            .map(|(x_i, y_i)| *x_i * y_i + (F::one() - x_i) * (F::one() - y_i))
            .product()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x == y).into()
    }

    fn to_lookup_index(&self) -> u64 {
        interleave_bits(self.0 as u32, self.1 as u32)
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for both 32-bit and 64-bit
        (self.0 == self.1).into()
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
}

impl<const WORD_SIZE: usize, F: JoltField> SparseDenseSumcheckAlt<F> for BEQInstruction<WORD_SIZE> {
    const NUM_PREFIXES: usize = 1;
    const NUM_SUFFIXES: usize = 1;

    fn combine(prefixes: &[F], suffixes: &[F]) -> F {
        debug_assert_eq!(
            prefixes.len(),
            <Self as SparseDenseSumcheckAlt<F>>::NUM_PREFIXES
        );
        debug_assert_eq!(
            suffixes.len(),
            <Self as SparseDenseSumcheckAlt<F>>::NUM_SUFFIXES
        );
        prefixes[0] * suffixes[0]
    }

    fn update_prefix_checkpoints(checkpoints: &mut [Option<F>], r_x: F, r_y: F, j: usize) {
        let updated =
            checkpoints[0].unwrap_or(F::one()) * (r_x * r_y + (F::one() - r_x) * (F::one() - r_y));
        checkpoints[0] = Some(updated);
    }

    fn prefix_mle(
        _: usize,
        checkpoints: &[Option<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[0].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= r_x * y + (F::one() - r_x) * (F::one() - y);
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = F::from_u8(b.pop_msb());
            result *= x * y_msb + (F::one() - x) * (F::one() - y_msb);
        }
        let (x, y) = b.uninterleave();
        if x != y {
            return F::zero();
        }
        result
    }

    fn suffix_mle(_: usize, b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        (x == y).into()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{
        instruction_mle_test_large, instruction_mle_test_small,
        jolt::instruction::{test::prefix_suffix_test, JoltInstruction},
        jolt_instruction_test,
    };

    use super::BEQInstruction;

    #[test]
    fn beq_prefix_suffix() {
        prefix_suffix_test::<Fr, BEQInstruction<32>>();
    }

    instruction_mle_test_small!(beq_mle_small, BEQInstruction<8>);
    instruction_mle_test_large!(beq_mle_large, BEQInstruction<32>);

    #[test]
    fn beq_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = BEQInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Test edge-cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            BEQInstruction::<WORD_SIZE>(100, 0),
            BEQInstruction::<WORD_SIZE>(0, 100),
            BEQInstruction::<WORD_SIZE>(1, 0),
            BEQInstruction::<WORD_SIZE>(0, u32_max),
            BEQInstruction::<WORD_SIZE>(u32_max, 0),
            BEQInstruction::<WORD_SIZE>(u32_max, u32_max),
            BEQInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            BEQInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn beq_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = BEQInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Test edge-cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            BEQInstruction::<WORD_SIZE>(100, 0),
            BEQInstruction::<WORD_SIZE>(0, 100),
            BEQInstruction::<WORD_SIZE>(1, 0),
            BEQInstruction::<WORD_SIZE>(0, u64_max),
            BEQInstruction::<WORD_SIZE>(u64_max, 0),
            BEQInstruction::<WORD_SIZE>(u64_max, u64_max),
            BEQInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            BEQInstruction::<WORD_SIZE>(1 << 32, u64_max),
            BEQInstruction::<WORD_SIZE>(1 << 63, 1),
            BEQInstruction::<WORD_SIZE>(1, 1 << 63),
            BEQInstruction::<WORD_SIZE>(u64_max - 1, u64_max),
            BEQInstruction::<WORD_SIZE>(u64_max, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
