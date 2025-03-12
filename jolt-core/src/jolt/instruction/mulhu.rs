use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::suffixes::Suffixes;
use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::instruction::suffixes::one::OneSuffix;
use crate::jolt::instruction::suffixes::upper_word::UpperWordSuffix;
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::subprotocols::sparse_dense_shout::{
    current_suffix_len, LookupBits, SparseDenseSumcheckAlt,
};
use crate::utils::instruction_utils::{
    assert_valid_parameters, concatenate_lookups, multiply_and_chunk_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MULHUInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for MULHUInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, M: usize) -> F {
        concatenate_lookups(vals, vals.len(), log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert_eq!(C * log2(M) as usize, 2 * WORD_SIZE);
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(0..C / 2),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        multiply_and_chunk_operands(self.0 as u128, self.1 as u128, C, log_M)
    }

    fn eta(&self) -> usize {
        1
    }

    fn subtable_entry(&self, _: usize, index: u64) -> u64 {
        index >> WORD_SIZE
    }

    fn to_lookup_index(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => self.0 * self.1,
            32 => self.0 * self.1,
            // 64 => (self.0 as u128) + (self.1 as u128),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn lookup_entry(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0).wrapping_mul(self.1) >> 8,
            32 => (self.0).wrapping_mul(self.1) >> 32,
            64 => ((self.0 as u128).wrapping_mul(self.1 as u128) >> 64) as u64,
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
        if j >= WORD_SIZE {
            return F::zero();
        }
        let d_j = F::from_u32(1 << (WORD_SIZE - 1 - j));
        // (r_j - b_j) * d_j
        if b_j == 1 {
            r_j * d_j - d_j
        } else {
            debug_assert_eq!(b_j, 0);
            r_j * d_j
        }
    }

    fn subtable_mle<F: JoltField>(&self, _: usize, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            result += F::from_u64(1 << (WORD_SIZE - 1 - i)) * r[i];
        }
        result
    }
}

impl<const WORD_SIZE: usize, F: JoltField> SparseDenseSumcheckAlt<WORD_SIZE, F>
    for MULHUInstruction<WORD_SIZE>
{
    const NUM_PREFIXES: usize = 1;

    fn combine(prefixes: &[F], suffixes: &[F]) -> F {
        prefixes[0] * suffixes[0] + suffixes[1]
    }

    fn suffixes() -> Vec<Suffixes<WORD_SIZE>> {
        vec![
            Suffixes::One(OneSuffix),
            Suffixes::UpperWord(UpperWordSuffix),
        ]
    }

    fn update_prefix_checkpoints(checkpoints: &mut [Option<F>], r_x: F, r_y: F, j: usize) {
        if j >= WORD_SIZE {
            return;
        }
        let x_shift = WORD_SIZE - j;
        let y_shift = WORD_SIZE - j - 1;
        let updated = checkpoints[0].unwrap_or(F::zero())
            + F::from_u64(1 << x_shift) * r_x
            + F::from_u64(1 << y_shift) * r_y;
        checkpoints[0] = Some(updated);
    }

    fn prefix_mle(
        _: usize,
        checkpoints: &[Option<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        mut j: usize,
    ) -> F {
        let mut result = checkpoints[0].unwrap_or(F::zero());
        if j >= WORD_SIZE {
            return result;
        }

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = WORD_SIZE - j;
            let y_shift = WORD_SIZE - j - 1;
            result += F::from_u64(1 << x_shift) * r_x;
            result += F::from_u64(1 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = WORD_SIZE - j - 1;
            let y_shift = WORD_SIZE - j - 2;
            result += F::from_u64(1 << x_shift) * x;
            result += F::from_u64(1 << y_shift) * F::from_u8(y_msb);
        }

        let suffix_len = current_suffix_len(2 * WORD_SIZE, j);
        if suffix_len > WORD_SIZE {
            result += F::from_u64(u64::from(b) << (suffix_len - WORD_SIZE));
        } else {
            println!("j={j} b.len={}, suffix_len={suffix_len}", b.len());
            let (b_high, _) = b.split(WORD_SIZE - suffix_len);
            result += F::from_u64(u64::from(b_high));
        }

        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::MULHUInstruction;
    use crate::{
        instruction_mle_test_large, instruction_mle_test_small, instruction_update_function_test,
        jolt::instruction::{test::prefix_suffix_test, JoltInstruction},
        jolt_instruction_test,
    };

    #[test]
    fn mulhu_prefix_suffix() {
        prefix_suffix_test::<Fr, MULHUInstruction<32>>();
    }

    instruction_mle_test_small!(mulhu_mle_small, MULHUInstruction<8>);
    instruction_mle_test_large!(mulhu_mle_large, MULHUInstruction<32>);
    instruction_update_function_test!(mulhu_update_fn, MULHUInstruction<32>);

    #[test]
    fn mulhu_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = MULHUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            MULHUInstruction::<WORD_SIZE>(100, 0),
            MULHUInstruction::<WORD_SIZE>(0, 100),
            MULHUInstruction::<WORD_SIZE>(1, 0),
            MULHUInstruction::<WORD_SIZE>(0, u32_max),
            MULHUInstruction::<WORD_SIZE>(u32_max, 0),
            MULHUInstruction::<WORD_SIZE>(u32_max, u32_max),
            MULHUInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            MULHUInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn mulhu_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = MULHUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            MULHUInstruction::<WORD_SIZE>(100, 0),
            MULHUInstruction::<WORD_SIZE>(0, 100),
            MULHUInstruction::<WORD_SIZE>(1, 0),
            MULHUInstruction::<WORD_SIZE>(0, u64_max),
            MULHUInstruction::<WORD_SIZE>(u64_max, 0),
            MULHUInstruction::<WORD_SIZE>(u64_max, u64_max),
            MULHUInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            MULHUInstruction::<WORD_SIZE>(1 << 32, u64_max),
            MULHUInstruction::<WORD_SIZE>(1 << 63, 1),
            MULHUInstruction::<WORD_SIZE>(1, 1 << 63),
            MULHUInstruction::<WORD_SIZE>(u64_max - 1, 1),
            MULHUInstruction::<WORD_SIZE>(1, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
