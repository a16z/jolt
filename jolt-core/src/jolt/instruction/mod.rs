use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;

use crate::utils;
use crate::jolt::subtable::LassoSubtable;

#[enum_dispatch]
pub trait JoltInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F;
  fn g_poly_degree(&self, C: usize) -> usize;
  fn subtables<F: PrimeField>(&self, C: usize) -> Vec<Box<dyn LassoSubtable<F>>>;
  fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;
  fn lookup_entry<F: PrimeField>(&self, C: usize, M: usize) -> F {
    let log_M = ark_std::log2(M) as usize;

    let subtable_lookup_indices = self.to_indices(C, ark_std::log2(M) as usize);

    let subtable_lookup_values: Vec<F> = self
      .subtables::<F>(C)
      .iter()
      .flat_map(|subtable| {
          subtable_lookup_indices.iter().map(|&lookup_index| {
              subtable.evaluate_mle(&utils::index_to_field_bitvector(lookup_index, log_M))
          })
      })
      .collect();

    self.combine_lookups(&subtable_lookup_values, C, M)
  }
}

pub trait Opcode {
  fn to_opcode(&self) -> u8 {
    unsafe { *<*const _>::from(self).cast::<u8>() }
  }
}

pub mod add;
pub mod and;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod blt;
pub mod bltu;
pub mod bne;
pub mod jal;
pub mod jalr;
pub mod or;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod sub;
pub mod xor;

#[cfg(test)]
pub mod test;
