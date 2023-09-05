use ark_ff::PrimeField;
use std::marker::PhantomData;

use super::{ChunkIndices, JoltInstruction, SubtableDecomposition};
use crate::jolt::subtable::{eq::EQSubtable, LassoSubtable};

#[derive(Copy, Clone, Default)]
pub struct EQInstruction(u64, u64);

impl<F: PrimeField> JoltInstruction<F> for EQInstruction {
  fn combine_lookups(vals: &[F]) -> F {
    unimplemented!("TODO");
  }

  fn g_poly_degree() -> usize {
    unimplemented!("TODO");
  }
}

impl SubtableDecomposition for EQInstruction {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(EQSubtable::new())]
  }
}

impl ChunkIndices for EQInstruction {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C] {
    unimplemented!("TODO");
  }
}
