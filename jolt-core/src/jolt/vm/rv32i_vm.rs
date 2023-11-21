use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::{instruction_lookups::InstructionLookups, Jolt};
use crate::jolt::instruction::{
  add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
  bgeu::BGEUInstruction, blt::BLTInstruction, bltu::BLTUInstruction, bne::BNEInstruction,
  jal::JALInstruction, jalr::JALRInstruction, or::ORInstruction, sll::SLLInstruction,
  slt::SLTInstruction, sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction,
  sub::SUBInstruction, xor::XORInstruction, JoltInstruction, Opcode,
};
use crate::jolt::subtable::{
  and::AndSubtable, eq::EqSubtable, eq_abs::EqAbsSubtable, eq_msb::EqMSBSubtable,
  gt_msb::GtMSBSubtable, identity::IdentitySubtable, lt_abs::LtAbsSubtable, ltu::LtuSubtable,
  or::OrSubtable, sll::SllSubtable, sra_sign::SraSignSubtable, srl::SrlSubtable,
  truncate_overflow::TruncateOverflowSubtable, xor::XorSubtable, zero_lsb::ZeroLSBSubtable,
  LassoSubtable,
};

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
  RV32I,
  // ADD: ADDInstruction,
  // AND: ANDInstruction,
  // BEQ: BEQInstruction,
  // BGE: BGEInstruction,
  // BGEU: BGEUInstruction,
  // BLT: BLTInstruction,
  // BLTU: BLTUInstruction,
  BNE: BNEInstruction,
  // JAL: JALInstruction,
  // JALR: JALRInstruction,
  // OR: ORInstruction,
  // SLL: SLLInstruction,
  // SLT: SLTInstruction,
  // SLTU: SLTUInstruction,
  // SRA: SRAInstruction,
  // SRL: SRLInstruction,
  // SUB: SUBInstruction,
  XOR: XORInstruction
);
subtable_enum!(
  RV32ISubtables,
  // AND: AndSubtable<F>,
  // EQ_ABS: EqAbsSubtable<F>,
  // EQ_MSB: EqMSBSubtable<F>,
  EQ: EqSubtable<F>,
  // GT_MSB: GtMSBSubtable<F>,
  // IDENTITY: IdentitySubtable<F>,
  // LT_ABS: LtAbsSubtable<F>,
  // LTU: LtuSubtable<F>,
  // OR: OrSubtable<F>,
  // SLL0: SllSubtable<F, 0>,
  // SLL1: SllSubtable<F, 1>,
  // SLL2: SllSubtable<F, 2>,
  // SLL3: SllSubtable<F, 3>,
  // SRA_SIGN: SraSignSubtable<F>,
  // SRL0: SrlSubtable<F, 0>,
  // SRL1: SrlSubtable<F, 1>,
  // SRL2: SrlSubtable<F, 2>,
  // SRL3: SrlSubtable<F, 3>,
  // TRUNCATE: TruncateOverflowSubtable<F>,
  XOR: XorSubtable<F>
  // ZERO_LSB: ZeroLSBSubtable<F>
);

// ==================== JOLT ====================

pub enum RV32IJoltVM {}

const C: usize = 4;
const M: usize = 1 << 16;

impl<F, G> Jolt<F, G, C, M> for RV32IJoltVM
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type InstructionSet = RV32I;
  type Subtables = RV32ISubtables<F>;
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
  use crate::jolt::instruction::bgeu::BGEUInstruction;
  use crate::jolt::instruction::jal::JALInstruction;
  use crate::jolt::instruction::jalr::JALRInstruction;
  use crate::jolt::instruction::sub::SUBInstruction;
  use crate::jolt::instruction::{JoltInstruction, Opcode};
  use crate::jolt::subtable::and::AndSubtable;
  use crate::jolt::subtable::eq::EqSubtable;
  use crate::jolt::subtable::identity::IdentitySubtable;
  use crate::jolt::subtable::ltu::LtuSubtable;
  use crate::jolt::subtable::truncate_overflow::TruncateOverflowSubtable;
  use crate::jolt::subtable::zero_lsb::ZeroLSBSubtable;
  use crate::jolt::subtable::LassoSubtable;
  use crate::jolt::vm::instruction_lookups::InstructionLookupsProof;
  use crate::{
    jolt::vm::rv32i_vm::{
      BNEInstruction, InstructionLookups, Jolt, RV32IJoltVM, XORInstruction, C, M, RV32I,
    },
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
    let ops: Vec<RV32I> = vec![
      RV32I::XOR(XORInstruction(420, 69)),
      RV32I::XOR(XORInstruction(420, 420)),
      // RV32I::XOR(XORInstruction(420, 69)),
      RV32I::BNE(BNEInstruction(82, 200)),
      RV32I::BNE(BNEInstruction(82, 200)),
    ];

    let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
    let mut prover_transcript = Transcript::new(b"example");
    let proof: InstructionLookupsProof<Fr, EdwardsProjective> =
      RV32IJoltVM::prove_instruction_lookups(ops, r.clone(), &mut prover_transcript);
    let mut verifier_transcript = Transcript::new(b"example");
    assert!(RV32IJoltVM::verify_instruction_lookups(proof, r, &mut verifier_transcript).is_ok());
  }

  #[test]
  fn instruction_set_subtables() {
    let mut subtable_set: HashSet<_> = HashSet::new();
    for instruction in <RV32IJoltVM as Jolt<_, EdwardsProjective, C, M>>::InstructionSet::iter() {
      for subtable in instruction.subtables::<Fr>(C) {
        // panics if subtable cannot be cast to enum variant
        let _ = <RV32IJoltVM as Jolt<_, EdwardsProjective, C, M>>::Subtables::from(
          subtable.subtable_id(),
        );
        subtable_set.insert(subtable.subtable_id());
      }
    }
    assert_eq!(
      subtable_set.len(),
      <RV32IJoltVM as Jolt<_, EdwardsProjective, C, M>>::Subtables::COUNT,
      "Unused enum variants in Subtables"
    );
  }

  // #[test]
  // fn subtable_reuse() {
  //   // Setup VM / Instructions / Subtables
  //   #[repr(u8)]
  //   #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
  //   #[enum_dispatch(JoltInstruction)]
  //   pub enum ReuseInstructionSet {
  //     BNE(BNEInstruction),
  //     BGEU(BGEUInstruction),
  //   }
  //   impl Opcode for ReuseInstructionSet {}

  //   subtable_enum!(
  //     ReuseSubtables,
  //     EQ: EqSubtable<F>,
  //     LT: LtuSubtable<F>
  //   );

  //   let ops: Vec<ReuseInstructionSet> = vec![
  //     ReuseInstructionSet::BNE(BNEInstruction(100, 100)),
  //     ReuseInstructionSet::BGEU(BGEUInstruction(100, 100)),
  //   ];

  //   pub enum ReuseJoltVM {}

  //   const C: usize = 8;
  //   const M: usize = 1 << 16;

  //   impl<F, G> Jolt<F, G, C, M> for ReuseJoltVM
  //   where
  //     F: PrimeField,
  //     G: CurveGroup<ScalarField = F>,
  //   {
  //     type InstructionSet = ReuseInstructionSet;
  //     type Subtables = ReuseSubtables<F>;
  //   }

  //   // e2e test

  //   let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
  //   let mut prover_transcript = Transcript::new(b"example");
  //   let instruction_lookups_proof: JoltProof<EdwardsProjective> =
  //     ReuseJoltVM::prove_lookups(ops, r.clone(), &mut prover_transcript);
  //   let mut verifier_transcript = Transcript::new(b"example");
  //   assert!(ReuseJoltVM::verify(instruction_lookups_proof, &r, &mut verifier_transcript).is_ok());
  // }

  // #[test]
  // fn subtable_reuse_increased_complication() {
  //   // Setup VM / Instructions / Subtables
  //   #[repr(u8)]
  //   #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
  //   #[enum_dispatch(JoltInstruction)]
  //   pub enum ReuseInstructionSet {
  //     AND(ANDInstruction),
  //     SUB(SUBInstruction),
  //     JAL(JALInstruction),
  //     JALR(JALRInstruction),
  //   }
  //   impl Opcode for ReuseInstructionSet {}

  //   #[repr(usize)]
  //   #[enum_dispatch(LassoSubtable<F>)]
  //   #[derive(EnumCountMacro, EnumIter)]
  //   pub enum ReuseCSubtables<F: PrimeField> {
  //     AND(AndSubtable<F>),
  //     ID(IdentitySubtable<F>),
  //     TRUNC(TruncateOverflowSubtable<F>),
  //     ZERO(ZeroLSBSubtable<F>),
  //   }
  //   impl<F: PrimeField> From<TypeId> for ReuseCSubtables<F> {
  //     fn from(subtable_id: TypeId) -> Self {
  //       if subtable_id == TypeId::of::<AndSubtable<F>>() {
  //         ReuseCSubtables::<F>::from(AndSubtable::new())
  //       } else if subtable_id == TypeId::of::<IdentitySubtable<F>>() {
  //         ReuseCSubtables::<F>::from(IdentitySubtable::new())
  //       } else if subtable_id == TypeId::of::<TruncateOverflowSubtable<F>>() {
  //         ReuseCSubtables::<F>::from(TruncateOverflowSubtable::new())
  //       } else if subtable_id == TypeId::of::<ZeroLSBSubtable<F>>() {
  //         ReuseCSubtables::<F>::from(ZeroLSBSubtable::new())
  //       } else {
  //         panic!("Unexpected subtable id {:?}", subtable_id)
  //       }
  //     }
  //   }

  //   impl<F: PrimeField> Into<usize> for ReuseCSubtables<F> {
  //     fn into(self) -> usize {
  //       unsafe { *<*const _>::from(&self).cast::<usize>() }
  //     }
  //   }

  //   let ops: Vec<ReuseInstructionSet> = vec![
  //     ReuseInstructionSet::AND(ANDInstruction(100, 100)),
  //     ReuseInstructionSet::SUB(SUBInstruction(100, 100)),
  //     ReuseInstructionSet::JAL(JALInstruction(100, 100)),
  //     ReuseInstructionSet::JALR(JALRInstruction(100, 100)),
  //   ];

  //   pub enum ReuseJoltVM {}

  //   const C: usize = 8;
  //   const M: usize = 1 << 16;

  //   impl<F, G> Jolt<F, G, C, M> for ReuseJoltVM
  //   where
  //     F: PrimeField,
  //     G: CurveGroup<ScalarField = F>,
  //   {
  //     type InstructionSet = ReuseInstructionSet;
  //     type Subtables = ReuseCSubtables<F>;
  //   }

  //   // e2e test

  //   let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
  //   let mut prover_transcript = Transcript::new(b"example");
  //   let instruction_lookups_proof: JoltProof<EdwardsProjective> =
  //     ReuseJoltVM::prove_lookups(ops, r.clone(), &mut prover_transcript);
  //   let mut verifier_transcript = Transcript::new(b"example");
  //   assert!(ReuseJoltVM::verify(instruction_lookups_proof, &r, &mut verifier_transcript).is_ok());
  // }
}
