use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;

use crate::jolt::vm::test_vm::TestSubtables;

#[enum_dispatch]
pub trait LassoSubtable<F: PrimeField>: 'static {
  fn subtable_id(&self) -> TypeId {
    TypeId::of::<Self>()
  }
  fn materialize(&self, M: usize) -> Vec<F>;
  fn evaluate_mle(&self, point: &[F]) -> F;
}

pub mod and;
pub mod eq;
pub mod eq_abs;
pub mod eq_msb;
pub mod gt_msb;
pub mod lt_abs;
pub mod ltu;
pub mod or;
pub mod xor;

#[cfg(test)]
pub mod test;
