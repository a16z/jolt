pub use super::slt::SLTInstruction;

pub use SLTInstruction as BLTInstruction;

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::{test_rng, One, Zero};
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::BLTInstruction;

  #[test]
  fn blt_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 8;
    const M: usize = 1 << 16;

    for _ in 0..256 {
      let x = rng.next_u64() as i64;
      let y = rng.next_u64() as i64;
      jolt_instruction_test!(BLTInstruction(x as u64, y as u64), (x < y).into());
    }
    for _ in 0..256 {
      let x = rng.next_u64() as i64;
      jolt_instruction_test!(BLTInstruction(x as u64, x as u64), Fr::zero());
    }
  }
}
