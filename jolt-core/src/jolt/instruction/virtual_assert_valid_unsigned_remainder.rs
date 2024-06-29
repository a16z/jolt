use crate::{field::JoltField, jolt::subtable::right_is_zero::RightIsZeroSubtable};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::JoltInstruction;
use crate::{
    jolt::{
        instruction::SubtableIndices,
        subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct AssertValidUnsignedRemainderInstruction(pub u64, pub u64);

impl JoltInstruction for AssertValidUnsignedRemainderInstruction {
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

        for i in 0..C {
            sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        // LTU(r, y) + EQ(y, 0)
        sum + divisor_is_zero
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
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C)),
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
        let remainder = self.0;
        let divisor = self.1;
        (divisor == 0 || remainder < divisor).into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(rng.next_u32() as u64, rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::AssertValidUnsignedRemainderInstruction;

    #[test]
    fn assert_valid_unsigned_remainder_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = AssertValidUnsignedRemainderInstruction(x, y);
            jolt_instruction_test!(instruction);
        }
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            jolt_instruction_test!(AssertValidUnsignedRemainderInstruction(x, x));
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            AssertValidUnsignedRemainderInstruction(100, 0),
            AssertValidUnsignedRemainderInstruction(0, 100),
            AssertValidUnsignedRemainderInstruction(1, 0),
            AssertValidUnsignedRemainderInstruction(0, u32_max),
            AssertValidUnsignedRemainderInstruction(u32_max, 0),
            AssertValidUnsignedRemainderInstruction(u32_max, u32_max),
            AssertValidUnsignedRemainderInstruction(u32_max, 1 << 8),
            AssertValidUnsignedRemainderInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
