use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::Jolt;
use crate::jolt::instruction::bge::BGEInstruction;
use crate::jolt::instruction::bgeu::BGEUInstruction;
use crate::jolt::instruction::{
  bne::BNEInstruction, sll::SLLInstruction, xor::XORInstruction, JoltInstruction, Opcode,
};
use crate::jolt::subtable::ltu::LtuSubtable;
use crate::jolt::subtable::{eq::EqSubtable, sll::SllSubtable, xor::XorSubtable, LassoSubtable};

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

instruction_set!(
  TestInstructionSet,
  XOR: XORInstruction,
  BNE: BNEInstruction,
  SLL: SLLInstruction
);
subtable_enum!(
  TestSubtables,
  XOR: XorSubtable<F>,
  EQ: EqSubtable<F>,
  SLL0: SllSubtable<F, 0>,
  SLL1: SllSubtable<F, 1>,
  SLL2: SllSubtable<F, 2>,
  SLL3: SllSubtable<F, 3>,
  SLL4: SllSubtable<F, 4>,
  SLL5: SllSubtable<F, 5>
);

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
  use ark_ec::CurveGroup;
  use ark_ff::PrimeField;
  use ark_std::{log2, test_rng, One, Zero};
  use enum_dispatch::enum_dispatch;
  use merlin::Transcript;
  use rand_chacha::rand_core::RngCore;
  use std::collections::HashSet;

  use crate::jolt::instruction::and::ANDInstruction;
use crate::jolt::instruction::bge::BGEInstruction;
  use crate::jolt::instruction::bgeu::BGEUInstruction;
  use crate::jolt::instruction::jal::JALInstruction;
use crate::jolt::instruction::jalr::JALRInstruction;
use crate::jolt::instruction::sll::SLLInstruction;
  use crate::jolt::instruction::sub::SUBInstruction;
use crate::jolt::instruction::{JoltInstruction, Opcode};
  use crate::jolt::subtable::and::AndSubtable;
use crate::jolt::subtable::eq::EqSubtable;
  use crate::jolt::subtable::identity::IdentitySubtable;
use crate::jolt::subtable::ltu::LtuSubtable;
  use crate::jolt::subtable::LassoSubtable;
  use crate::jolt::subtable::truncate_overflow::TruncateOverflowSubtable;
use crate::jolt::subtable::zero_lsb::ZeroLSBSubtable;
use crate::{
    jolt::vm::test_vm::{BNEInstruction, Jolt, TestInstructionSet, TestJoltVM, XORInstruction},
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{index_to_field_bitvector, math::Math, random::RandomTape, split_bits},
  };
  use std::any::TypeId;
  use strum::{EnumCount, IntoEnumIterator};
  use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

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
      TestInstructionSet::BNE(BNEInstruction(82, 200)),
      TestInstructionSet::BNE(BNEInstruction(82, 200)),
      // TestInstructionSet::XOR(XORInstruction(420, 69)),
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

  #[test]
  fn subtable_reuse() {
    // Setup VM / Instructions / Subtables
    #[repr(u8)]
    #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
    #[enum_dispatch(JoltInstruction)]
    pub enum ReuseInstructionSet {
      BNE(BNEInstruction),
      BGEU(BGEUInstruction),
    }
    impl Opcode for ReuseInstructionSet {}

    subtable_enum!(
      ReuseSubtables,
      EQ: EqSubtable<F>,
      LT: LtuSubtable<F>
    );

    let ops: Vec<ReuseInstructionSet> = vec![
      ReuseInstructionSet::BNE(BNEInstruction(100, 100)),
      ReuseInstructionSet::BGEU(BGEUInstruction(100, 100)),
    ];

    pub enum ReuseJoltVM {}

    impl<F: PrimeField, G: CurveGroup<ScalarField = F>> Jolt<F, G> for ReuseJoltVM {
      const MEMORY_OPS_PER_STEP: usize = 4;
      const C: usize = 4;
      const M: usize = 1 << 16;

      type InstructionSet = ReuseInstructionSet;
      type Subtables = ReuseSubtables<F>;
    }

    // e2e test
 
    let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
    let mut prover_transcript = Transcript::new(b"example");
    let proof = <ReuseJoltVM as Jolt<_, EdwardsProjective>>::prove_lookups(
      ops,
      r.clone(),
      &mut prover_transcript,
    );
    let mut verifier_transcript = Transcript::new(b"example");
    assert!(<ReuseJoltVM as Jolt<_, EdwardsProjective>>::verify(
      proof,
      &r,
      &mut verifier_transcript
    )
    .is_ok());
  }

  #[test]
  fn subtable_reuse_increased_complication() {
    // Setup VM / Instructions / Subtables
    #[repr(u8)]
    #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
    #[enum_dispatch(JoltInstruction)]
    pub enum ReuseInstructionSet {
      AND(ANDInstruction),
      SUB(SUBInstruction),
      JAL(JALInstruction),
      JALR(JALRInstruction)
    }
    impl Opcode for ReuseInstructionSet {}

    #[repr(usize)]
    #[enum_dispatch(LassoSubtable<F>)]
    #[derive(EnumCountMacro, EnumIter)]
    pub enum ReuseCSubtables<F: PrimeField> { 
      AND(AndSubtable<F>),
      ID(IdentitySubtable<F>),
      TRUNC(TruncateOverflowSubtable<F>),
      ZERO(ZeroLSBSubtable<F>)
    }
    impl<F: PrimeField> From<TypeId> for ReuseCSubtables<F> {
      fn from(subtable_id: TypeId) -> Self {
        if subtable_id == TypeId::of::<AndSubtable<F>>() {
          ReuseCSubtables::<F>::from(AndSubtable::new())
        } else if subtable_id == TypeId::of::<IdentitySubtable<F>>() {
          ReuseCSubtables::<F>::from(IdentitySubtable::new())
        } else if subtable_id == TypeId::of::<TruncateOverflowSubtable<F>>() {
          ReuseCSubtables::<F>::from(TruncateOverflowSubtable::new())
        } else if subtable_id == TypeId::of::<ZeroLSBSubtable<F>>() {
          ReuseCSubtables::<F>::from(ZeroLSBSubtable::new())
        } else { 
          panic!("Unexpected subtable id {:?}", subtable_id) 
        }
      }
    }

    impl<F: PrimeField> Into<usize> for ReuseCSubtables<F> {
      fn into(self) -> usize {
        unsafe { *<*const _>::from(&self).cast::<usize>() }
      }
    }

    let ops: Vec<ReuseInstructionSet> = vec![
      ReuseInstructionSet::AND(ANDInstruction(100, 100)),
      ReuseInstructionSet::SUB(SUBInstruction(100, 100)),
      ReuseInstructionSet::JAL(JALInstruction(100, 100)),
      ReuseInstructionSet::JALR(JALRInstruction(100, 100)),
    ];

    pub enum ReuseJoltVM {}

    impl<F: PrimeField, G: CurveGroup<ScalarField = F>> Jolt<F, G> for ReuseJoltVM {
      const MEMORY_OPS_PER_STEP: usize = 4;
      const C: usize = 8;
      const M: usize = 1 << 16;

      type InstructionSet = ReuseInstructionSet;
      type Subtables = ReuseCSubtables<F>;
    }

    // e2e test
 
    let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
    let mut prover_transcript = Transcript::new(b"example");
    let proof = <ReuseJoltVM as Jolt<_, EdwardsProjective>>::prove_lookups(
      ops,
      r.clone(),
      &mut prover_transcript,
    );
    let mut verifier_transcript = Transcript::new(b"example");
    assert!(<ReuseJoltVM as Jolt<_, EdwardsProjective>>::verify(
      proof,
      &r,
      &mut verifier_transcript
    )
    .is_ok());
  }
}
