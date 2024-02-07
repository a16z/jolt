use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::JoltInstruction;
use crate::{
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, eq_msb::EqMSBSubtable, gt_msb::GtMSBSubtable,
        lt_abs::LtAbsSubtable, ltu::LtuSubtable, LassoSubtable,
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct BLTInstruction(pub u64, pub u64);

impl JoltInstruction for BLTInstruction {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, _: usize) -> F {
        debug_assert!(vals.len() % C == 0);
        let mut vals_by_subtable = vals.chunks_exact(C);

        let gt_msb = vals_by_subtable.next().unwrap();
        let eq_msb = vals_by_subtable.next().unwrap();
        let ltu = vals_by_subtable.next().unwrap();
        let eq = vals_by_subtable.next().unwrap();
        let lt_abs = vals_by_subtable.next().unwrap();
        let eq_abs = vals_by_subtable.next().unwrap();

        // Accumulator for LTU(x_{<s}, y_{<s})
        let mut ltu_sum = lt_abs[0];
        // Accumulator for EQ(x_{<s}, y_{<s})
        let mut eq_prod = eq_abs[0];
        for i in 1..C {
            ltu_sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }

        // x_s * (1 - y_s) + EQ(x_s, y_s) * LTU(x_{<s}, y_{<s})
        gt_msb[0] + eq_msb[0] * ltu_sum
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C + 1
    }

    fn subtables<F: PrimeField>(&self, _: usize) -> Vec<Box<dyn LassoSubtable<F>>> {
        vec![
            Box::new(GtMSBSubtable::new()),
            Box::new(EqMSBSubtable::new()),
            Box::new(LtuSubtable::new()),
            Box::new(EqSubtable::new()),
            Box::new(LtAbsSubtable::new()),
            Box::new(EqAbsSubtable::new()),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0 as u64, self.1 as u64, C, log_M)
    }

    fn lookup_entry_u64(&self) -> u64 {
        (self.0 < self.1).into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand_core::RngCore;
        Self(rng.next_u32() as u64, rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;
    use ark_std::{test_rng, One, Zero};
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::BLTInstruction;

    #[test]
    fn blt_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u64() as i64;
            let y = rng.next_u64() as i64;
            jolt_instruction_test!(BLTInstruction(x as u64, y as u64), (x < y).into());
            assert_eq!(
                BLTInstruction(x as u64, y as u64).lookup_entry::<Fr>(C, M),
                (x < y).into()
            );
        }
        for _ in 0..256 {
            let x = rng.next_u64() as i64;
            jolt_instruction_test!(BLTInstruction(x as u64, x as u64), Fr::zero());
            assert_eq!(
                BLTInstruction(x as u64, x as u64).lookup_entry::<Fr>(C, M),
                Fr::zero()
            );
        }
    }
}
