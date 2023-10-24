pub use super::sltu::SLTUInstruction;

pub use SLTUInstruction as BLTUInstruction;

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use ark_std::{test_rng, One};
  use rand_chacha::rand_core::RngCore;

  use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

  use super::BLTUInstruction;

  #[test]
  fn bltu_instruction_e2e() {
    let mut rng = test_rng();
    const C: usize = 8;
    const M: usize = 1 << 16;

    for _ in 0..256 {
      let (x, y) = (rng.next_u64(), rng.next_u64());
      jolt_instruction_test!(BLTUInstruction(x, y), (x < y).into());
    }
  }
}
