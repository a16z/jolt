use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{or::OrSubtable, LassoSubtable};
use crate::subprotocols::sparse_dense_shout::{LookupBits, SparseDenseSumcheckAlt};
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};
use crate::utils::{interleave_bits, uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ORInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for ORInstruction<WORD_SIZE> {
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
        vec![(Box::new(OrSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn eta(&self) -> usize {
        1
    }

    fn subtable_entry(&self, _: usize, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x | y) as u64
    }

    fn to_lookup_index(&self) -> u64 {
        interleave_bits(self.0 as u32, self.1 as u32)
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for both 32-bit and 64-bit word sizes
        self.0 | self.1
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
        l: usize,
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
        l: usize,
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
            shift * (r_j - x) * (F::one() - y)
        } else {
            // Update y_{j/2} to r_j
            let x = r_prev.unwrap();
            let y = F::from_u8(b_j);
            shift * (F::one() - x) * (r_j - y)
        }
    }

    fn subtable_mle<F: JoltField>(&self, _: usize, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (WORD_SIZE - 1 - i)) * (x_i + y_i - x_i * y_i);
        }
        result
    }
}

impl<const WORD_SIZE: usize, F: JoltField> SparseDenseSumcheckAlt<F> for ORInstruction<WORD_SIZE> {
    const NUM_PREFIXES: usize = 1;
    const NUM_SUFFIXES: usize = 2;

    fn combine(prefixes: &[F], suffixes: &[F]) -> F {
        debug_assert_eq!(
            prefixes.len(),
            <Self as SparseDenseSumcheckAlt<F>>::NUM_PREFIXES
        );
        debug_assert_eq!(
            suffixes.len(),
            <Self as SparseDenseSumcheckAlt<F>>::NUM_SUFFIXES
        );
        prefixes[0] * suffixes[0] + suffixes[1]
    }

    fn update_prefix_checkpoints(checkpoints: &mut [Option<F>], r_x: F, r_y: F, j: usize) {
        let shift = WORD_SIZE - 1 - j / 2;
        let updated = checkpoints[0].unwrap_or(F::zero())
            + F::from_u32(1 << shift) * (r_x + r_y - (r_x * r_y));
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
        let mut result = checkpoints[0].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(1 << shift) * (r_x + y - (r_x * y));
        } else {
            let y_msb = b.pop_msb() as u32;
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(c + y_msb - c * y_msb) * F::from_u32(1 << shift);
        }
        let (x, y) = b.uninterleave();
        let suffix_len = WORD_SIZE - j / 2 - 1 - b.len() / 2;
        result += F::from_u32((u32::from(x) | u32::from(y)) << suffix_len);

        result
    }

    fn suffix_mle(l: usize, b: LookupBits) -> u32 {
        match l {
            0 => 1,
            1 => {
                let (x, y) = b.uninterleave();
                u32::from(x) | u32::from(y)
            }
            _ => unimplemented!("Unexpected value l={l}"),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{
        instruction_mle_test_large, instruction_mle_test_small, instruction_update_function_test,
        jolt::instruction::{test::prefix_suffix_test, JoltInstruction},
        jolt_instruction_test,
        subprotocols::sparse_dense_shout::SparseDenseSumcheckAlt,
    };

    use super::ORInstruction;

    #[test]
    fn or_prefix_suffix() {
        prefix_suffix_test::<Fr, ORInstruction<32>>();
    }

    instruction_mle_test_small!(or_mle_small, ORInstruction<8>);
    instruction_mle_test_large!(or_mle_large, ORInstruction<32>);
    instruction_update_function_test!(or_update_fn, ORInstruction<32>);

    #[test]
    fn or_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let y = rng.next_u32() as u64;
            let instruction = ORInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ORInstruction::<WORD_SIZE>(100, 0),
            ORInstruction::<WORD_SIZE>(0, 100),
            ORInstruction::<WORD_SIZE>(1, 0),
            ORInstruction::<WORD_SIZE>(0, u32_max),
            ORInstruction::<WORD_SIZE>(u32_max, 0),
            ORInstruction::<WORD_SIZE>(u32_max, u32_max),
            ORInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            ORInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn or_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let instruction = ORInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            ORInstruction::<WORD_SIZE>(100, 0),
            ORInstruction::<WORD_SIZE>(0, 100),
            ORInstruction::<WORD_SIZE>(1, 0),
            ORInstruction::<WORD_SIZE>(0, u64_max),
            ORInstruction::<WORD_SIZE>(u64_max, 0),
            ORInstruction::<WORD_SIZE>(u64_max, u64_max),
            ORInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            ORInstruction::<WORD_SIZE>(1 << 8, u64_max),
            ORInstruction::<WORD_SIZE>(u64_max, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
