use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use std::fmt::Debug;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::Jolt;
use crate::jolt::instruction::{
  eq::EQInstruction, xor::XORInstruction, ChunkIndices, Opcode, SubtableDecomposition,
};
use crate::jolt::subtable::{eq::EQSubtable, xor::XORSubtable, LassoSubtable};

macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[repr(u8)]
        #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
        #[enum_dispatch(ChunkIndices, SubtableDecomposition)]
        pub enum $enum_name { $($alias($struct)),+ }
        impl Opcode for $enum_name {}
    };
}

macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCountMacro, EnumIter, Debug)]
        pub enum $enum_name<F: PrimeField> { $($alias($struct)),+ }
        impl<F: PrimeField> From<TypeId> for $enum_name<F> {
          fn from(subtable_id: TypeId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id") }
          }
        }
    };
}

instruction_set!(TestInstructionSet, XOR: XORInstruction, EQ: EQInstruction);
subtable_enum!(TestSubtables, XOR: XORSubtable<F>, EQ: EQSubtable<F>);

// ==================== JOLT ====================

pub enum TestJoltVM {}

impl<F: PrimeField> Jolt<F> for TestJoltVM {
  const C: usize = 4;
  const M: usize = 1 << 16;

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
    jolt::vm::test_vm::{EQInstruction, Jolt, TestInstructionSet, TestJoltVM, XORInstruction},
    utils::{index_to_field_bitvector, random::RandomTape, split_bits},
  };

  #[test]
  fn e2e() {
    <TestJoltVM as Jolt<Fr>>::prove(vec![
      TestInstructionSet::XOR(XORInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 69)),
      TestInstructionSet::EQ(EQInstruction(420, 420)),
    ]);
  }

  // TODO(moodlezoup): test that union of VM::InstructionSet's subtables = VM::Subtables
}
