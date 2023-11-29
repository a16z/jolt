use ark_ff::PrimeField;

use super::JoltInstruction;
use crate::jolt::subtable::{
  identity::IdentitySubtable, srl::SrlSubtable,
  LassoSubtable,
};
use crate::utils::instruction_utils::chunk_and_concatenate_for_shift;

#[derive(Copy, Clone, Default, Debug)]
pub struct SRLInstruction(pub u64, pub u64);

impl JoltInstruction for SRLInstruction {
  fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
    assert!(C <= 10);
    assert!(vals.len() == C * C);

    let mut subtable_vals = vals.chunks_exact(C);
    let mut vals_filtered: Vec<F> = Vec::with_capacity(C);
    for i in 0..C {
      let subtable_val = subtable_vals.next().unwrap();
      vals_filtered.extend_from_slice(&subtable_val[i..i + 1]);
    }

    vals_filtered.iter().sum()
  }

  fn g_poly_degree(&self, _: usize) -> usize {
    1
  }

  fn subtables<F: PrimeField>(&self, C: usize) -> Vec<Box<dyn LassoSubtable<F>>> {
    let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
      Box::new(SrlSubtable::<F, 0>::new()),
      Box::new(SrlSubtable::<F, 1>::new()),
      Box::new(SrlSubtable::<F, 2>::new()),
      Box::new(SrlSubtable::<F, 3>::new()),
      Box::new(SrlSubtable::<F, 4>::new()),
      Box::new(SrlSubtable::<F, 5>::new()),
      Box::new(SrlSubtable::<F, 6>::new()),
      Box::new(SrlSubtable::<F, 7>::new()),
      Box::new(SrlSubtable::<F, 8>::new()),
      Box::new(SrlSubtable::<F, 9>::new()),
    ];
    subtables.truncate(C);
    subtables.reverse();
    subtables
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

  use super::SRLInstruction;

  #[test]
  fn srl_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 6;
    const M: usize = 1 << 22;

    for _ in 0..8 {
      let (x, y) = (rng.next_u64(), rng.next_u64());

      let entry: u64 = x.checked_shr((y % 64) as u32).unwrap_or(0);

      jolt_instruction_test!(SRLInstruction(x, y), entry.into());
      assert_eq!(SRLInstruction(x, y).lookup_entry::<Fr>(C, M), entry.into());
    }
  }
}
