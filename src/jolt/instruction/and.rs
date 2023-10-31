use std::marker::PhantomData;

use ark_ff::PrimeField;
use ark_std::log2;
use typenum::Unsigned;

use super::JoltInstruction;
use crate::jolt::subtable::{and::AndSubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};
use crate::utils::math::Math;

#[derive(Copy, Clone, Default, Debug)]
pub struct ANDInstruction<F: PrimeField, C: Unsigned, M: Unsigned> {
  pub rs1: u64,
  pub rs2: u64,
  _field: PhantomData<F>,
  _C: PhantomData<C>,
  _M: PhantomData<M>,
}

impl<F: PrimeField, C: Unsigned, M: Unsigned> ANDInstruction<F, C, M> {
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

impl<F: PrimeField, C: Unsigned, M: Unsigned> JoltInstruction<F, C, M> for ANDInstruction<F, C, M> {
  fn combine_lookups(&self, vals: &[F]) -> F {
    concatenate_lookups(vals, C::to_usize(), M::to_usize().log_2() / 2)
  }

  fn g_poly_degree(&self) -> usize {
    1
  }

  fn subtables(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(AndSubtable::new())]
  }

  fn to_indices(&self) -> Vec<usize> {
    chunk_and_concatenate_operands(self.rs1, self.rs2, C::to_usize(), M::to_usize().log_2())
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::test_rng;
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::ANDInstruction;

  #[test]
  fn and_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 8;
    const M: usize = 1 << 16;

    for _ in 0..256 {
      let (x, y) = (rng.next_u64(), rng.next_u64());
      jolt_instruction_test!(ANDInstruction(x, y), (x & y).into());
    }
  }
}
