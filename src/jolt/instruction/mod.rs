use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use typenum::Unsigned;

use crate::jolt::{subtable::LassoSubtable, vm::test_vm::TestInstructionSet};

#[enum_dispatch]
pub trait JoltInstruction<F: PrimeField, C: Unsigned, M: Unsigned> {
  fn combine_lookups(&self, vals: &[F]) -> F;
  fn g_poly_degree(&self) -> usize;
  fn subtables(&self) -> Vec<Box<dyn LassoSubtable<F>>>;
  fn to_indices(&self) -> Vec<usize>;
}

pub trait Opcode {
  fn to_opcode(&self) -> u8 {
    unsafe { *<*const _>::from(self).cast::<u8>() }
  }
}

// pub mod add;
pub mod and;
// pub mod beq;
// pub mod bge;
// pub mod bgeu;
// pub mod blt;
// pub mod bltu;
// pub mod bne;
// pub mod jal;
// pub mod jalr;
// pub mod or;
pub mod sll;
// pub mod slt;
// pub mod sltu;
// pub mod sub;
// pub mod xor;

#[cfg(test)]
pub mod test;
