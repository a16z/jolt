use std::marker::PhantomData;

use ark_ff::PrimeField;
use ark_std::log2;
use typenum::{Sub1, Unsigned, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9};

use super::JoltInstruction;
use crate::jolt::subtable::{sll::SllSubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_and_concatenate_for_shift, concatenate_lookups};
use crate::utils::math::Math;

#[derive(Copy, Clone, Default, Debug)]
pub struct SLLInstruction<F: PrimeField, C: Unsigned, M: Unsigned> {
  pub rs1: u64,
  pub rs2: u64,
  _field: PhantomData<F>,
  _C: PhantomData<C>,
  _M: PhantomData<M>,
}

impl<F: PrimeField, C: Unsigned, M: Unsigned> SLLInstruction<F, C, M> {
  pub fn new(rs1: u64, rs2: u64) -> Self {
    Self {
      rs1,
      rs2,
      _field: PhantomData,
      _C: PhantomData,
      _M: PhantomData,
    }
  }
}

impl<F: PrimeField, C: Unsigned, M: Unsigned> JoltInstruction<F, C, M> for SLLInstruction<F, C, M> {
  fn combine_lookups(&self, vals: &[F]) -> F {
    let C = C::to_usize();
    assert!(vals.len() == C * C);

    let mut subtable_vals = vals.chunks_exact(C);
    let mut vals_filtered: Vec<F> = Vec::with_capacity(C);
    for i in 0..C {
      let subtable_val = subtable_vals.next().unwrap();
      vals_filtered.extend_from_slice(&subtable_val[i..i + 1]);
    }

    concatenate_lookups(&vals_filtered, C, ((M::to_usize()) / 2).log_2() as usize)
  }

  fn g_poly_degree(&self) -> usize {
    1
  }

  fn subtables(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
      Box::new(SllSubtable::<F, U0>::new()),
      Box::new(SllSubtable::<F, U1>::new()),
      Box::new(SllSubtable::<F, U2>::new()),
      Box::new(SllSubtable::<F, U3>::new()),
      Box::new(SllSubtable::<F, U4>::new()),
      Box::new(SllSubtable::<F, U5>::new()),
      Box::new(SllSubtable::<F, U6>::new()),
      Box::new(SllSubtable::<F, U7>::new()),
      Box::new(SllSubtable::<F, U8>::new()),
      Box::new(SllSubtable::<F, U9>::new()),
    ];
    subtables.truncate(C::to_usize());
    subtables.reverse();
    subtables
  }

  fn to_indices(&self) -> Vec<usize> {
    chunk_and_concatenate_for_shift(self.rs1, self.rs2, C::to_usize(), M::to_usize().log_2())
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::test_rng;
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::SLLInstruction;

  #[test]
  fn sll_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 6;
    const M: usize = 1 << 22;

    for _ in 0..8 {
      let (x, y) = (rng.next_u64(), rng.next_u64());

      let entry: u64 = x.checked_shl((y % 64) as u32).unwrap_or(0);

      jolt_instruction_test!(SLLInstruction(x, y), entry.into());
    }
  }
}
