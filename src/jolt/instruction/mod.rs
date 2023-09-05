use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;

use crate::jolt::{subtable::LassoSubtable, vm::test_vm::TestInstructionSet};

trait JoltInstruction<F: PrimeField> {
  // TODO: C, M

  fn combine_lookups(vals: &[F]) -> F;
  fn g_poly_degree() -> usize;
}

#[enum_dispatch]
pub trait SubtableDecomposition {
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>>;
}

#[enum_dispatch]
pub trait ChunkIndices {
  fn to_indices<const C: usize, const LOG_M: usize>(&self) -> [usize; C];
}

pub trait Opcode {
  fn to_opcode(&self) -> u8 {
    unsafe { *<*const _>::from(self).cast::<u8>() }
  }
}

pub mod eq;
pub mod xor;
