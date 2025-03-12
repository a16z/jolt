use std::cmp::min;

use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{sll::SllSubtable, LassoSubtable};
use crate::subprotocols::sparse_dense_shout::{LookupBits, SparseDenseSumcheckAlt};
use crate::utils::instruction_utils::{
    assert_valid_parameters, chunk_and_concatenate_for_shift, concatenate_lookups,
};
use crate::utils::math::Math;
use crate::utils::{interleave_bits, uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SLLInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SLLInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(C <= 10);
        concatenate_lookups(vals, C, (log2(M) / 2) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // We have to pre-define subtables in this way because `CHUNK_INDEX` needs to be a constant,
        // i.e. known at compile time (so we cannot do a `map` over the range of `C`,
        // which only happens at runtime).
        let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
            Box::new(SllSubtable::<F, 0, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 1, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 2, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 3, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 4, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 5, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 6, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 7, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 8, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 9, WORD_SIZE>::new()),
        ];
        subtables.truncate(C);
        subtables.reverse();

        let indices = (0..C).map(SubtableIndices::from);
        subtables.into_iter().zip(indices).collect()
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        // \sum_{k = 0}^{2^b - 1} eq(y, bin(k)) * (\sum_{j = 0}^{m'-1} 2^{k + j} * x_{b - j - 1}),
        // where m = min(b, max( 0, (k + b * (CHUNK_INDEX + 1)) - WORD_SIZE))
        // and m' = b - m

        // We assume the first half is chunk(X_i) and the second half is always chunk(Y_0)
        debug_assert!(r.len() % 2 == 0);

        let log_WORD_SIZE = log2(WORD_SIZE) as usize;

        let b = r.len() / 2;
        let x: Vec<_> = r.iter().step_by(2).collect();
        let y: Vec<_> = r.iter().skip(1).step_by(2).collect();

        let mut result = F::zero();

        // min with 1 << b is included for test cases with subtables of bit-length smaller than 6
        for k in 0..min(WORD_SIZE, 1 << b) {
            // bit-decompose k
            let k_bits = k
                .get_bits(log_WORD_SIZE)
                .iter()
                .map(|bit| F::from_u64(*bit as u64))
                .collect::<Vec<F>>(); // big-endian

            // Compute eq(y, bin(k))
            let mut eq_term = F::one();
            // again, min with b is included when subtables of bit-length less than 6 are used
            for i in 0..min(log_WORD_SIZE, b) {
                eq_term *= k_bits[log_WORD_SIZE - 1 - i] * y[b - 1 - i]
                    + (F::one() - k_bits[log_WORD_SIZE - 1 - i]) * (F::one() - y[b - 1 - i]);
            }

            let m = if (k + b) > WORD_SIZE {
                min(b, (k + b) - WORD_SIZE)
            } else {
                0
            };
            let m_prime = b - m;

            // Compute \sum_{j = 0}^{m'-1} 2^{k + j} * x_{b - j - 1}
            let shift_x_by_k = (0..m_prime)
                .enumerate()
                .map(|(j, _)| F::from_u64(1_u64 << (j + k)) * x[b - 1 - j])
                .fold(F::zero(), |acc, val| acc + val);

            result += eq_term * shift_x_by_k;
        }
        result
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let shift = y % WORD_SIZE as u32;
        if WORD_SIZE != 64 {
            ((x as u64) << shift) % (1u64 << WORD_SIZE)
        } else {
            (x as u64) << shift
        }
    }

    fn to_lookup_index(&self) -> u64 {
        interleave_bits(self.0 as u32, self.1 as u32)
    }

    fn lookup_entry(&self) -> u64 {
        // SLL is specified to ignore all but the last 5 (resp. 6) bits of y: https://jemu.oscc.cc/SLL
        if WORD_SIZE == 32 {
            (self.0 as u32)
                .checked_shl(self.1 as u32 % WORD_SIZE as u32)
                .unwrap_or(0)
                .into()
        } else if WORD_SIZE == 64 {
            self.0
                .checked_shl((self.1 % WORD_SIZE as u64) as u32)
                .unwrap_or(0)
        } else {
            panic!("SLL is only implemented for 32-bit or 64-bit word sizes")
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
}

impl<const WORD_SIZE: usize, F: JoltField> SparseDenseSumcheckAlt<F> for SLLInstruction<WORD_SIZE> {
    const NUM_PREFIXES: usize = WORD_SIZE * 3 / 4;
    const NUM_SUFFIXES: usize = 1 + WORD_SIZE * 3 / 4;

    fn combine(prefixes: &[F], suffixes: &[F]) -> F {
        debug_assert_eq!(
            prefixes.len(),
            <Self as SparseDenseSumcheckAlt<F>>::NUM_PREFIXES
        );
        debug_assert_eq!(
            suffixes.len(),
            <Self as SparseDenseSumcheckAlt<F>>::NUM_SUFFIXES
        );

        suffixes[0]
            + prefixes
                .iter()
                .zip(suffixes[1..].iter())
                .map(|(prefix, suffix)| *prefix * suffix)
                .sum::<F>()
    }

    fn update_prefix_checkpoints(checkpoints: &mut [Option<F>], r_x: F, _: F, j: usize) {
        checkpoints[j / 2] = Some(r_x);
    }

    fn prefix_mle(
        l: usize,
        checkpoints: &[Option<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        let x_variables_bound = j / 2;
        if l == x_variables_bound {
            if let Some(r_x) = r_x {
                r_x
            } else {
                F::from_u32(c)
            }
        } else if l < x_variables_bound {
            checkpoints[l].unwrap()
        } else {
            let (x, _) = b.uninterleave();
            let index = l - x_variables_bound - 1;
            if index >= x.len() {
                F::zero()
            } else {
                F::from_u8(x.get_bit(l - x_variables_bound - 1))
            }
        }
    }

    fn suffix_mle(l: usize, b: LookupBits) -> u32 {
        debug_assert!(l < <Self as SparseDenseSumcheckAlt<F>>::NUM_SUFFIXES);

        let (x, y) = b.uninterleave();
        let shift = y % WORD_SIZE;

        if l == 0 {
            u32::from(x) << shift
        } else {
            let x_index = l - 1;
            if (WORD_SIZE - 1 - x_index + shift) > (WORD_SIZE - 1) {
                0
            } else {
                1 << (WORD_SIZE - 1 - x_index + shift)
            }
        }
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

    use super::SLLInstruction;

    instruction_mle_test_small!(sll_mle_small, SLLInstruction<8>);
    instruction_mle_test_large!(sll_mle_large, SLLInstruction<32>);

    #[test]
    fn sll_prefix_suffix() {
        prefix_suffix_test::<Fr, SLLInstruction<32>>();
    }

    #[test]
    fn sll_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32(), rng.next_u32());
            let instruction = SLLInstruction::<WORD_SIZE>(x as u64, y as u64);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SLLInstruction::<WORD_SIZE>(100, 0),
            SLLInstruction::<WORD_SIZE>(0, 100),
            SLLInstruction::<WORD_SIZE>(1, 0),
            SLLInstruction::<WORD_SIZE>(0, u32_max),
            SLLInstruction::<WORD_SIZE>(u32_max, 0),
            SLLInstruction::<WORD_SIZE>(u32_max, u32_max),
            SLLInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            SLLInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    #[ignore]
    fn sll_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = SLLInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            SLLInstruction::<WORD_SIZE>(100, 0),
            SLLInstruction::<WORD_SIZE>(0, 100),
            SLLInstruction::<WORD_SIZE>(1, 0),
            SLLInstruction::<WORD_SIZE>(0, u64_max),
            SLLInstruction::<WORD_SIZE>(u64_max, 0),
            SLLInstruction::<WORD_SIZE>(u64_max, u64_max),
            SLLInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            SLLInstruction::<WORD_SIZE>(1 << 8, u64_max),
            SLLInstruction::<WORD_SIZE>(u64_max, 1 << 63),
            SLLInstruction::<WORD_SIZE>(1 << 63, 63),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
