use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;

use super::instruction::{eq::EQInstruction, xor::XORInstruction};
use super::subtable::LassoSubtable;
use crate::jolt::vm::test_vm::TestInstructionSet;

#[enum_dispatch]
pub trait JoltInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F;
  fn g_poly_degree(&self, C: usize) -> usize;
  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>>;
  fn to_indices(&self, C: usize, M: usize) -> Vec<usize>;
}

#[macro_export]
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[repr(u8)]
        #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name { $($alias($struct)),+ }
        impl Opcode for $enum_name {}
    };
}

pub trait Opcode {
  fn to_opcode(&self) -> u8 {
    unsafe { *<*const _>::from(self).cast::<u8>() }
  }
}

pub mod eq;
pub mod xor;
