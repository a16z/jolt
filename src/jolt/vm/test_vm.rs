use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use std::fmt::Debug;
use std::marker::PhantomData;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::Jolt;
use crate::jolt::instruction::{
  eq::EQInstruction, xor::XORInstruction, ChunkIndices, Opcode, SubtableDecomposition,
};
use crate::jolt::subtable::{eq::EQSubtable, xor::XORSubtable, LassoSubtable};

// ==================== INSTRUCTIONS ====================

#[repr(u8)]
#[derive(Copy, Clone, EnumIter, EnumCountMacro)]
#[enum_dispatch(ChunkIndices, SubtableDecomposition)]
pub enum TestInstructionSet {
  XOR(XORInstruction),
  EQ(EQInstruction),
}

impl Opcode for TestInstructionSet {}

// ==================== SUBTABLES ====================

#[enum_dispatch(LassoSubtable<F>)]
#[derive(EnumCountMacro, EnumIter, Debug)]
pub enum TestSubtables<F: PrimeField> {
  XOR(XORSubtable<F>),
  EQ(EQSubtable<F>),
}

// TODO(moodlezoup): make this a macro
impl<F: PrimeField> From<TypeId> for TestSubtables<F> {
  fn from(subtable_id: TypeId) -> Self {
    let xor_id = TypeId::of::<XORSubtable<F>>();
    let eq_id = TypeId::of::<EQSubtable<F>>();

    if subtable_id == TypeId::of::<XORSubtable<F>>() {
      TestSubtables::XOR(XORSubtable::new())
    } else if subtable_id == TypeId::of::<EQSubtable<F>>() {
      TestSubtables::EQ(EQSubtable::new())
    } else {
      panic!("Unexpected subtable id")
    }
  }
}

// ==================== JOLT ====================

pub enum TestJoltVM {}

impl<F: PrimeField> Jolt<F> for TestJoltVM {
  const C: usize = 4;
  const LOG_M: usize = 16;

  type InstructionSet = TestInstructionSet;
  type Subtables = TestSubtables<F>;
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
    jolt::test_vm::{EQInstruction, Jolt, TestInstructionSet, TestJoltVM, XORInstruction},
    utils::{index_to_field_bitvector, random::RandomTape, split_bits},
  };

  #[test]
  fn e2e() {
    TestJoltVM::<Fr>::prove(vec![
      TestInstructionSet::XOR(XORInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 420)),
    ]);
  }

  // TODO(moodlezoup): test that union of VM::InstructionSet's subtables = VM::Subtables
}
