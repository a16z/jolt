use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::Jolt;
use crate::jolt::instruction::{bne::BNEInstruction, xor::XORInstruction, JoltInstruction, Opcode};
use crate::jolt::subtable::{eq::EqSubtable, xor::XorSubtable, LassoSubtable};

macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[repr(u8)]
        #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name { $($alias($struct)),+ }
        impl Opcode for $enum_name {}
    };
}

// TODO(moodlezoup): Consider replacing From<TypeId> and Into<usize> with
//     combined trait/function to_enum_index(subtable: &dyn LassoSubtable<F>) => usize
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[repr(usize)]
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCountMacro, EnumIter)]
        pub enum $enum_name<F: PrimeField> { $($alias($struct)),+ }
        impl<F: PrimeField> From<TypeId> for $enum_name<F> {
          fn from(subtable_id: TypeId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) } // TODO(moodlezoup): better error handling
          }
        }

        impl<F: PrimeField> Into<usize> for $enum_name<F> {
          fn into(self) -> usize {
            unsafe { *<*const _>::from(&self).cast::<usize>() }
          }
        }
    };
}

instruction_set!(TestInstructionSet, XOR: XORInstruction, BNE: BNEInstruction);
subtable_enum!(TestSubtables, XOR: XorSubtable<F>, EQ: EqSubtable<F>);

// ==================== JOLT ====================

pub enum TestJoltVM {}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> Jolt<F, G> for TestJoltVM {
  const MEMORY_OPS_PER_STEP: usize = 4;
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
  use std::collections::HashSet;
  use strum::{EnumCount, IntoEnumIterator};

  use crate::jolt::instruction::JoltInstruction;
  use crate::{
    jolt::vm::test_vm::{BNEInstruction, Jolt, TestInstructionSet, TestJoltVM, XORInstruction},
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subprotocols::sumcheck::SumcheckInstanceProof,
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
      TestInstructionSet::XOR(XORInstruction(420, 420)),
      TestInstructionSet::XOR(XORInstruction(420, 69)),
      TestInstructionSet::XOR(XORInstruction(420, 69)),
      // TestInstructionSet::EQ(EQInstruction(420, 69)),
      // TestInstructionSet::EQ(EQInstruction(420, 420)),
      // TestInstructionSet::EQ(EQInstruction(420, 420)),
      // TestInstructionSet::EQ(EQInstruction(420, 420)),
    ];

    let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
    let mut prover_transcript = Transcript::new(b"example");
    let proof = <TestJoltVM as Jolt<_, EdwardsProjective>>::prove_lookups(
      ops,
      r.clone(),
      &mut prover_transcript,
    );
    let mut verifier_transcript = Transcript::new(b"example");
    assert!(<TestJoltVM as Jolt<_, EdwardsProjective>>::verify(
      proof,
      &r,
      &mut verifier_transcript
    )
    .is_ok());
  }

  #[test]
  fn instruction_set_subtables() {
    let mut subtable_set: HashSet<_> = HashSet::new();
    for instruction in <TestJoltVM as Jolt<_, EdwardsProjective>>::InstructionSet::iter() {
      for subtable in instruction.subtables::<Fr>() {
        // panics if subtable cannot be cast to enum variant
        let _ = <TestJoltVM as Jolt<_, EdwardsProjective>>::Subtables::from(subtable.subtable_id());
        subtable_set.insert(subtable.subtable_id());
      }
    }
    assert_eq!(
      subtable_set.len(),
      <TestJoltVM as Jolt<_, EdwardsProjective>>::Subtables::COUNT,
      "Unused enum variants in Subtables"
    );
  }
}
