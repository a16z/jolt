use crate::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::{
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable,
        ltu::LtuSubtable, right_msb::RightMSBSubtable, LassoSubtable,
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SLTInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SLTInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);

        let left_msb = vals_by_subtable[0];
        let right_msb = vals_by_subtable[1];
        let ltu = vals_by_subtable[2];
        let eq = vals_by_subtable[3];
        let lt_abs = vals_by_subtable[4];
        let eq_abs = vals_by_subtable[5];

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sum = lt_abs[0];
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prod = eq_abs[0];

        for i in 0..C - 2 {
            ltu_sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        // Do not need to update `eq_prod` for the last iteration
        ltu_sum += ltu[C - 2] * eq_prod;

        // x_s * (1 - y_s) + EQ(x_s, y_s) * LTU(x_{<s}, y_{<s})
        left_msb[0] * (F::one() - right_msb[0])
            + (left_msb[0] * right_msb[0] + (F::one() - left_msb[0]) * (F::one() - right_msb[0]))
                * ltu_sum
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
            (x < y) as u64
        } else if WORD_SIZE == 64 {
            let x = self.0 as i64;
            let y = self.1 as i64;
            (x < y) as u64
        } else {
            panic!("SLT is only implemented for 32-bit or 64-bit word sizes")
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        if WORD_SIZE == 32 {
            Self(rng.next_u32() as u64, rng.next_u32() as u64)
        } else if WORD_SIZE == 64 {
            Self(rng.next_u64(), rng.next_u64())
        } else {
            panic!("Only 32-bit and 64-bit word sizes are supported");
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::SLTInstruction;

    #[test]
    fn slt_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let y = rng.next_u32() as u64;
            let instruction = SLTInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SLTInstruction::<32>(100, 0),
            SLTInstruction::<32>(0, 100),
            SLTInstruction::<32>(1, 0),
            SLTInstruction::<32>(0, u32_max),
            SLTInstruction::<32>(u32_max, 0),
            SLTInstruction::<32>(u32_max, u32_max),
            SLTInstruction::<32>(u32_max, 1 << 8),
            SLTInstruction::<32>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn slt_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        for _ in 0..256 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let instruction = SLTInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        let i64_min = i64::MIN as u64;
        let i64_max = i64::MAX as u64;
        let instructions = vec![
            SLTInstruction::<64>(100, 0),
            SLTInstruction::<64>(0, 100),
            SLTInstruction::<64>(1, 0),
            SLTInstruction::<64>(0, i64_max),
            SLTInstruction::<64>(i64_max, 0),
            SLTInstruction::<64>(i64_max, i64_max),
            SLTInstruction::<64>(i64_max, 1 << 32),
            SLTInstruction::<64>(1 << 32, i64_max),
            SLTInstruction::<64>(i64_min, 0),
            SLTInstruction::<64>(0, i64_min),
            SLTInstruction::<64>(i64_min, i64_max),
            SLTInstruction::<64>(i64_max, i64_min),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
