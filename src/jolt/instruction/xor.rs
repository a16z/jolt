use ark_ff::PrimeField;
use std::marker::PhantomData;

use super::{ChunkIndices, JoltInstruction, SubtableDecomposition};
use crate::jolt::subtable::{xor::XORSubtable, LassoSubtable};

#[derive(Copy, Clone, Default)]
pub struct XORInstruction(u64, u64);

impl<F: PrimeField> JoltInstruction<F> for XORInstruction {
  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    1
  }
}

impl SubtableDecomposition for XORInstruction {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(XORSubtable::new())]
  }
}

impl ChunkIndices for XORInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}
