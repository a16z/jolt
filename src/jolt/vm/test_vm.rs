use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use crate::{instruction_set, subtable_enum};

use super::Jolt;
use crate::jolt::vm::instruction::{eq::EQInstruction, xor::XORInstruction, Opcode};
use crate::jolt::vm::subtable::{eq::EQSubtable, xor::XORSubtable};

subtable_enum!(TestSubtables, XOR: XORSubtable<F>, EQ: EQSubtable<F>);
instruction_set!(TestInstructionSet, XOR: XORInstruction, EQ: EQInstruction);

// ==================== JOLT ====================

pub enum TestJoltVM {}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> Jolt<F, G> for TestJoltVM {
  const C: usize = 4;
  const M: usize = 1 << 16;

  type InstructionSet = TestInstructionSet;
  type Subtables = TestSubtables<G::ScalarField>;
}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_ff::PrimeField;
  use ark_std::{log2, test_rng, One, Zero};
  use merlin::Transcript;
  use rand_chacha::rand_core::RngCore;

  use crate::{
    jolt::vm::test_vm::{EQInstruction, Jolt, TestInstructionSet, TestJoltVM, XORInstruction},
    utils::{index_to_field_bitvector, math::Math, random::RandomTape, split_bits},
  };

  pub fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
    let mut rng = test_rng();
    let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
    for _ in 0..memory_bits {
      r_i.push(F::rand(&mut rng));
    }
    r_i
  }

  #[test]
  fn e2e() {
    let ops: Vec<TestInstructionSet> = vec![
      TestInstructionSet::XOR(XORInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 420)),
    ];

    let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
    let mut prover_transcript = Transcript::new(b"example");
    <TestJoltVM as Jolt<_, EdwardsProjective>>::prove(
      vec![
        TestInstructionSet::XOR(XORInstruction(420, 69)),
        TestInstructionSet::EQ(EQInstruction(420, 69)),
        TestInstructionSet::EQ(EQInstruction(420, 420)),
      ],
      r,
      &mut prover_transcript,
    );
  }

  // TODO(moodlezoup): test that union of VM::InstructionSet's subtables = VM::Subtables
}
