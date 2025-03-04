use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{and::AndSubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};
use crate::utils::{interleave_bits, uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ANDInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for ANDInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(AndSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        assert_eq!(WORD_SIZE, 8);
        (0..1 << 16).map(|i| self.materialize_entry(i)).collect()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x & y) as u64
    }

    fn to_lookup_index(&self) -> u64 {
        interleave_bits(self.0 as u32, self.1 as u32)
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for 32-bit and 64-bit word sizes
        self.0 & self.1
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

    // m_\ell(r_j, j, b_j)
    fn multiplicative_update<F: JoltField>(
        &self,
        j: usize,
        r_j: F,
        b_j: u8,
        r_prev: Option<F>,
        b_next: Option<u8>,
    ) -> F {
        F::one()
    }

    // a_\ell(r_j, j, b_j)
    fn additive_update<F: JoltField>(
        &self,
        j: usize,
        r_j: F,
        b_j: u8,
        r_prev: Option<F>,
        b_next: Option<u8>,
    ) -> F {
        let shift = F::from_u32(1 << (WORD_SIZE - 1 - (j / 2)));
        if j % 2 == 0 {
            // Update x_{j/2} to r_j
            let x = F::from_u8(b_j);
            let y = F::from_u8(b_next.unwrap());
            shift * (r_j - x) * y
        } else {
            // Update y_{j/2} to r_j
            let x = r_prev.unwrap();
            let y = F::from_u8(b_j);
            shift * x * (r_j - y)
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (WORD_SIZE - 1 - i)) * x_i * y_i;
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use rand_chacha::rand_core::RngCore;

    use crate::{
        field::JoltField, jolt::instruction::JoltInstruction, jolt_instruction_test,
        utils::index_to_field_bitvector,
    };

    use super::ANDInstruction;

    #[test]
    fn and_mle_small() {
        const WORD_SIZE: usize = 8;
        let materialized = ANDInstruction::<WORD_SIZE>::default().materialize();
        for (i, entry) in materialized.iter().enumerate() {
            assert_eq!(
                Fr::from_u64(*entry),
                ANDInstruction::<WORD_SIZE>::default()
                    .evaluate_mle(&index_to_field_bitvector(i as u64, 16)),
                "MLE did not match materialized table at index {i}",
            );
        }
    }

    #[test]
    fn and_mle_large() {
        let mut rng = test_rng();
        const WORD_SIZE: usize = 32;

        for _ in 0..1000 {
            let index = rng.next_u64();
            assert_eq!(
                Fr::from_u64(ANDInstruction::<WORD_SIZE>::default().materialize_entry(index)),
                ANDInstruction::<WORD_SIZE>::default()
                    .evaluate_mle(&index_to_field_bitvector(index, 64)),
                "MLE did not match materialized table at index {index}",
            );
        }
    }

    #[test]
    fn and_update_functions() {
        let mut rng = test_rng();
        const WORD_SIZE: usize = 32;
        let and = ANDInstruction::<WORD_SIZE>::default();

        for _ in 0..1000 {
            let index = rng.next_u64();
            let mut t_parameters: Vec<Fr> = index_to_field_bitvector(index, 2 * WORD_SIZE);
            let mut r_prev = None;

            for j in 0..2 * WORD_SIZE {
                let r_j = Fr::random(&mut rng);
                let b_j = if t_parameters[j].is_zero() { 0 } else { 1 };

                let b_next = if j == 2 * WORD_SIZE - 1 {
                    None
                } else {
                    Some(t_parameters[j + 1].to_u64().unwrap() as u8)
                };

                let actual = and.multiplicative_update(j, r_j, b_j, r_prev, b_next)
                    * and.evaluate_mle(&t_parameters)
                    + and.additive_update(j, r_j, b_j, r_prev, b_next);

                t_parameters[j] = r_j;
                r_prev = Some(r_j);
                let expected = and.evaluate_mle(&t_parameters);

                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn and_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = ANDInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ANDInstruction::<WORD_SIZE>(100, 0),
            ANDInstruction::<WORD_SIZE>(0, 100),
            ANDInstruction::<WORD_SIZE>(1, 0),
            ANDInstruction::<WORD_SIZE>(0, u32_max),
            ANDInstruction::<WORD_SIZE>(u32_max, 0),
            ANDInstruction::<WORD_SIZE>(u32_max, u32_max),
            ANDInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            ANDInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn and_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = ANDInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            ANDInstruction::<WORD_SIZE>(100, 0),
            ANDInstruction::<WORD_SIZE>(0, 100),
            ANDInstruction::<WORD_SIZE>(1, 0),
            ANDInstruction::<WORD_SIZE>(0, u64_max),
            ANDInstruction::<WORD_SIZE>(u64_max, 0),
            ANDInstruction::<WORD_SIZE>(u64_max, u64_max),
            ANDInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            ANDInstruction::<WORD_SIZE>(1 << 32, u64_max),
            ANDInstruction::<WORD_SIZE>(1 << 63, 1),
            ANDInstruction::<WORD_SIZE>(1, 1 << 63),
            ANDInstruction::<WORD_SIZE>(u64_max - 1, 1),
            ANDInstruction::<WORD_SIZE>(1, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
