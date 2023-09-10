use ark_ff::PrimeField;
use ark_std::log2;

use super::{ChunkIndices, JoltInstruction, SubtableDecomposition};
use crate::jolt::subtable::{xor::XORSubtable, LassoSubtable};

#[derive(Copy, Clone, Default)]
pub struct XORInstruction(pub u64, pub u64);

impl<F: PrimeField> JoltInstruction<F> for XORInstruction {
  fn combine_lookups<const C: usize, const M: usize>(vals: &[F]) -> F {
    assert_eq!(vals.len(), C);
    let increment = log2(M) as usize;

    let mut sum = F::zero();
    for i in 0..C {
      let weight: u64 = 1u64 << (i * increment);
      sum += F::from(weight) * vals[i];
    }
    sum
  }

  fn g_poly_degree<const C: usize>() -> usize {
    1
  }
}

impl SubtableDecomposition for XORInstruction {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(XORSubtable::new())]
  }
}

impl ChunkIndices for XORInstruction {
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
