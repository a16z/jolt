use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;

use crate::jolt::{subtable::LassoSubtable, vm::test_vm::TestInstructionSet};

trait JoltInstruction<F: PrimeField> {
  fn combine_lookups<const C: usize, const M: usize>(vals: &[F]) -> F;
  fn g_poly_degree<const C: usize>() -> usize;
}

#[enum_dispatch]
pub trait SubtableDecomposition {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>>;
}

#[enum_dispatch]
pub trait ChunkIndices {
  fn to_indices(&self, C: usize, M: usize) -> Vec<usize>;
}

pub trait Opcode {
  fn to_opcode(&self) -> u8 {
    unsafe { *<*const _>::from(self).cast::<u8>() }
  }
}

pub mod eq;
pub mod xor;
