// use core::slice::SlicePattern;
use ark_ff::PrimeField;
use ark_std::log2;

use super::JoltInstruction;
use crate::jolt::subtable::{iden::IDENSubtable, lowerk::LOWERKSubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups, add_and_chunk_operands};

#[derive(Copy, Clone, Default, Debug)]
pub struct ADDInstruction(pub u64, pub u64);

impl JoltInstruction for ADDInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
    // The first C are from IDEN and the last C are from LOWER9
    assert!(vals.len() == 2 * C);

    // The output is the LOWER9(most significant chunk) || IDEN of other chunks
    concatenate_lookups([&vals[C..(C+1)], &vals[1..C]].concat().as_slice(), C, M)
  }

  fn g_poly_degree(&self, _: usize) -> usize {
    1
  }

  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![Box::new(IDENSubtable::new()), Box::new(LOWERKSubtable::new(9))]
  }

  fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
    add_and_chunk_operands(self.0, self.1, C, log_M)
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::test_rng;
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::ADDInstruction;

  #[test]
  fn add_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 8;
    const M: usize = 1 << 16;

    for _ in 0..256 {
      let (x, y) = (rng.next_u64(), rng.next_u64());
        jolt_instruction_test!(ADDInstruction(x, y), (x.overflowing_add(y)).0.into());
    }
  }
}
