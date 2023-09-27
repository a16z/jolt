use ark_ff::PrimeField;

use super::JoltInstruction;
use crate::jolt::subtable::{eq::EQSubtable, LassoSubtable};

#[derive(Copy, Clone, Default, Debug)]
pub struct EQInstruction(pub u64, pub u64);

impl JoltInstruction for EQInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], _: usize, _: usize) -> F {
    vals.iter().product()
  }

  fn g_poly_degree(&self, C: usize) -> usize {
    C
  }

  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(EQSubtable::new())]
  }

  fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
    let operand_bits: usize = log_M / 2;
    let operand_bit_mask: usize = (1 << operand_bits) - 1;
    (0..C)
      .map(|i| {
        let left: usize = (self.0 as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
        let right: usize = (self.1 as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
        (left << operand_bits) | right
      })
      .collect()
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::{test_rng, One};
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::EQInstruction;

  #[test]
  fn eq_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 8;
    const M: usize = 1 << 16;

    for _ in 0..256 {
      let (x, y) = (rng.next_u64(), rng.next_u64());
      jolt_instruction_test!(EQInstruction(x, y), (x == y).into());
    }
    for _ in 0..256 {
      let x = rng.next_u64();
      jolt_instruction_test!(EQInstruction(x, x), Fr::one());
    }
  }
}
