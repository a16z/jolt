use ark_ff::PrimeField;
use ark_std::log2;

use super::{ChunkIndices, JoltInstruction, SubtableDecomposition};
use crate::jolt::subtable::{eq::EQSubtable, LassoSubtable};

#[derive(Copy, Clone, Default)]
pub struct EQInstruction(u64, u64);

impl<F: PrimeField> JoltInstruction<F> for EQInstruction {
  fn combine_lookups<const C: usize, const M: usize>(vals: &[F]) -> F {
    vals.iter().product()
  }

  fn g_poly_degree<const C: usize>() -> usize {
    C
  }
}

impl SubtableDecomposition for EQInstruction {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(EQSubtable::new())]
  }
}

impl ChunkIndices for EQInstruction {
  fn to_indices<const C: usize, const M: usize>(&self) -> [usize; C] {
    let operand_bits: usize = (log2(M) / 2) as usize;
    let operand_bit_mask: usize = (1 << operand_bits) - 1;
    std::array::from_fn(|i| {
      let left: usize = (self.0 as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
      let right: usize = (self.1 as usize >> ((C - i - 1) * operand_bits)) & operand_bit_mask;
      (left << operand_bits) | right
    })
  }
}
