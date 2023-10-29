// use core::slice::SlicePattern;
use ark_ff::PrimeField;
use ark_std::log2;

use super::JoltInstruction;
use crate::jolt::subtable::{
  identity::IdentitySubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
  sll::SllSubtable,
};
use crate::utils::instruction_utils::{
  chunk_and_concatenate_for_shift, concatenate_lookups,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct SLLInstruction(pub u64, pub u64);

impl JoltInstruction for SLLInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
    assert!(vals.len() == 6 * C);

    let mut vals_by_subtable = vals.chunks_exact(C);

    concatenate_lookups(
      [
        &vals_by_subtable.next().unwrap()[0..1],
        &vals_by_subtable.next().unwrap()[1..2],
        &vals_by_subtable.next().unwrap()[2..3],
        &vals_by_subtable.next().unwrap()[3..4],
        &vals_by_subtable.next().unwrap()[4..5],
        &vals_by_subtable.next().unwrap()[5..6],
      ]
      .concat()
      .as_slice(),
      C,
      (log2(M)/2) as usize,
    )
  }

  fn g_poly_degree(&self, _: usize) -> usize {
    1
  }

  fn subtables<F: PrimeField>(&self) -> Vec<Box<dyn LassoSubtable<F>>> {
    vec![
      Box::new(SllSubtable::new(5)),
      Box::new(SllSubtable::new(4)),
      Box::new(SllSubtable::new(3)),
      Box::new(SllSubtable::new(2)),
      Box::new(SllSubtable::new(1)),
      Box::new(SllSubtable::new(0)),
    ]
  }

  fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
    chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
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

    for _ in 0..256 {
      let (x, y) = (rng.next_u64(), rng.next_u64());

      let entry: u64 = x.checked_shl((y%64) as u32).unwrap_or(0);

      jolt_instruction_test!(SLLInstruction(x, y), entry.into());
    }
  }
}
