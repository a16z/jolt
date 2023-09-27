use ark_ff::PrimeField;
use ark_std::log2;

use super::JoltInstruction;
use crate::jolt::subtable::{xor::XORSubtable, LassoSubtable};

#[derive(Copy, Clone, Default, Debug)]
pub struct XORInstruction(pub u64, pub u64);

impl JoltInstruction for XORInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
    assert_eq!(vals.len(), C);

    let mut sum = F::zero();
    let mut weight = F::one();
    let shift = F::from(1u64 << (log2(M) / 2));
    for i in 0..C {
      sum += weight * vals[C - i - 1];
      weight *= shift;
    }
    sum
  }

  fn g_poly_degree(&self, _: usize) -> usize {
    1
  }
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(XORSubtable::new())]
  }

  fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
    let operand_bits: usize = log_M / 2;
    let operand_bit_mask: usize = (1 << operand_bits) - 1;
    (0..C)
      .map(|i| {
        let left = (self.0 as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
        let right = (self.1 as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
        (left << operand_bits) | right
      })
      .collect()
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::test_rng;
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::XORInstruction;

  #[test]
  fn xor_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 8;
    const M: usize = 1 << 16;

    for _ in 0..256 {
      let (x, y) = (rng.next_u64(), rng.next_u64());
      jolt_instruction_test!(XORInstruction(x, y), (x ^ y).into());
    }
  }
}
