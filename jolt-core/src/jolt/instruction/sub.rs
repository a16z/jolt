// use core::slice::SlicePattern;
use ark_ff::PrimeField;
use ark_std::log2;

use super::JoltInstruction;
use crate::jolt::subtable::{
  identity::IdentitySubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
};
use crate::utils::instruction_utils::{
  add_and_chunk_operands, concatenate_lookups,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct SUBInstruction(pub u64, pub u64);

impl JoltInstruction for SUBInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
    // The first C are from IDEN and the last C are from LOWER9
    assert!(vals.len() == 2 * C);

    const WORD_SIZE: usize = 64;
    let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;

    let mut vals_by_subtable = vals.chunks_exact(C);
    let identity = vals_by_subtable.next().unwrap();
    let truncate_overflow = vals_by_subtable.next().unwrap();

    // The output is the LOWER9(most significant chunk) || IDEN of other chunks
    concatenate_lookups(
      [
        &truncate_overflow[0..=msb_chunk_index],
        &identity[msb_chunk_index + 1..C],
      ]
      .concat()
      .as_slice(),
      C,
      log2(M) as usize,
    )
  }

  fn g_poly_degree(&self, _: usize) -> usize {
    1
  }

  fn subtables<F: PrimeField>(&self, _: usize) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![
      Box::new(IdentitySubtable::new()),
      Box::new(TruncateOverflowSubtable::new()),
    ]
  }

  fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
    add_and_chunk_operands(self.0 as u128, (1u128 << 64) - self.1 as u128, C, log_M)
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::test_rng;
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::SUBInstruction;

  #[test]
  fn sub_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 8;
    const M: usize = 1 << 16;

    for _ in 0..256 {
      let (x, y) = (rng.next_u64(), rng.next_u64());
      jolt_instruction_test!(SUBInstruction(x, y), (x.overflowing_sub(y)).0.into());
    }
  }
}
