use ark_ff::PrimeField;

use super::JoltInstruction;
use crate::jolt::vm::subtable::{eq::EQSubtable, LassoSubtable};

#[derive(Copy, Clone, Default)]
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
