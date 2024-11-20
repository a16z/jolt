use crate::{field::JoltField, jolt::subtable::right_is_zero::RightIsZeroSubtable};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::{
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, left_is_zero::LeftIsZeroSubtable,
        left_msb::LeftMSBSubtable, lt_abs::LtAbsSubtable, ltu::LtuSubtable,
        right_msb::RightMSBSubtable, LassoSubtable,
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (remainder, divisor)
pub struct AssertValidSignedRemainderInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for AssertValidSignedRemainderInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);

        let left_msb = vals_by_subtable[0];
        let right_msb = vals_by_subtable[1];
        let eq = vals_by_subtable[2];
        let ltu = vals_by_subtable[3];
        let eq_abs = vals_by_subtable[4];
        let lt_abs = vals_by_subtable[5];
        let remainder_is_zero: F = vals_by_subtable[6].iter().product();
        let divisor_is_zero: F = vals_by_subtable[7].iter().product();

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sum = lt_abs[0];
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prod = eq_abs[0];

        for (ltu_i, eq_i) in ltu.iter().zip(eq) {
            ltu_sum += *ltu_i * eq_prod;
            eq_prod *= *eq_i;
        }

        // (1 - x_s - y_s) * LTU(x_{<s}, y_{<s}) + x_s * y_s * (1 - EQ(x_{<s}, y_{<s})) + (1 - x_s) * y_s * EQ(x, 0) + EQ(y, 0)
        (F::one() - left_msb[0] - right_msb[0]) * ltu_sum
            + left_msb[0] * right_msb[0] * (F::one() - eq_prod)
            + (F::one() - left_msb[0]) * right_msb[0] * remainder_is_zero
            + divisor_is_zero
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
            (Box::new(LeftMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(RightMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(EqAbsSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtAbsSubtable::new()), SubtableIndices::from(0)),
            (
                Box::new(LeftIsZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
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
        match WORD_SIZE {
            32 => {
                let remainder = self.0 as u32 as i32;
                let divisor = self.1 as u32 as i32;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 31;
                    let divisor_sign = divisor >> 31;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            64 => {
                let remainder = self.0 as i64;
                let divisor = self.1 as i64;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 63;
                    let divisor_sign = divisor >> 63;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
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

    use super::AssertValidSignedRemainderInstruction;

    #[test]
    fn assert_valid_signed_remainder_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let y = rng.next_u32() as u64;
            let instruction = AssertValidSignedRemainderInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = AssertValidSignedRemainderInstruction::<WORD_SIZE>(x, x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let i32_min: u64 = i32::MIN as u32 as u64;
        let instructions = vec![
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(100, 0),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(0, 100),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(1, 0),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(0, u32_max),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u32_max, 0),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u32_max, u32_max),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(1 << 8, u32_max),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(4294967295, 3909118204),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(i32_min, i32_min),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn assert_valid_signed_remainder_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = AssertValidSignedRemainderInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = AssertValidSignedRemainderInstruction::<WORD_SIZE>(x, x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let i64_min: u64 = i64::MIN as u64;
        let instructions = vec![
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(100, 0),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(0, 100),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(1, 0),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(0, u64_max),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u64_max, 0),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u64_max, u64_max),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(1 << 8, u64_max),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u64_max, 1 << 40),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(u64_max, u64_max - 1),
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(i64_min, i64_min),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
