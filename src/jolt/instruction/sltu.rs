// use ark_ff::PrimeField;

// use super::JoltInstruction;
// use crate::{
//   jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
//   utils::instruction_utils::chunk_and_concatenate_operands,
// };

// #[derive(Copy, Clone, Default, Debug)]
// pub struct SLTUInstruction(pub u64, pub u64);

// impl JoltInstruction for SLTUInstruction {
//   fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, _: usize) -> F {
//     let mut sum = F::zero();
//     let mut eq_prod = F::one();

//     for i in 0..C {
//       sum += vals[i] * eq_prod;
//       eq_prod *= vals[C + i];
//     }
//     sum
//   }

//   fn g_poly_degree(&self, C: usize) -> usize {
//     C
//   }

//   fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
//     vec![Box::new(LtuSubtable::new()), Box::new(EqSubtable::new())]
//   }

//   fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
//     chunk_and_concatenate_operands(self.0, self.1, C, log_M)
//   }
// }

// #[cfg(test)]
// mod test {
//   use ark_curve25519::Fr;
//   use ark_std::{test_rng, One, Zero};
//   use rand_chacha::rand_core::RngCore;

//   use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

//   use super::SLTUInstruction;

//   #[test]
//   fn sltu_instruction_e2e() {
//     let mut rng = test_rng();
//     const C: usize = 8;
//     const M: usize = 1 << 16;

//     for _ in 0..256 {
//       let (x, y) = (rng.next_u64(), rng.next_u64());
//       jolt_instruction_test!(SLTUInstruction(x, y), (x < y).into());
//     }
//     for _ in 0..256 {
//       let x = rng.next_u64();
//       jolt_instruction_test!(SLTUInstruction(x, x), Fr::zero());
//     }
//   }
// }
