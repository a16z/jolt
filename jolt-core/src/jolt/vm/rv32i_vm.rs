use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::{instruction_lookups::InstructionLookups, Jolt};
use crate::jolt::instruction::add::ADD32Instruction;
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

const WORD_SIZE: usize = 32;

instruction_set!(
  RV32I,
  ADD: ADD32Instruction,
  AND: ANDInstruction,
  BEQ: BEQInstruction,
  BGE: BGEInstruction,
  BGEU: BGEUInstruction,
  BLT: BLTInstruction,
  BLTU: BLTUInstruction,
  BNE: BNEInstruction,
  JAL: JALInstruction<WORD_SIZE>,
  JALR: JALRInstruction<WORD_SIZE>,
  OR: ORInstruction,
  SLL: SLLInstruction<WORD_SIZE>,
  // SLT: SLTInstruction,
  // SLTU: SLTUInstruction,
  SRA: SRAInstruction<WORD_SIZE>,
  SRL: SRLInstruction<WORD_SIZE>,
  SUB: SUBInstruction<WORD_SIZE>,
  XOR: XORInstruction
);
subtable_enum!(
  RV32ISubtables,
  AND: AndSubtable<F>,
  EQ_ABS: EqAbsSubtable<F>,
  EQ_MSB: EqMSBSubtable<F>,
  EQ: EqSubtable<F>,
  GT_MSB: GtMSBSubtable<F>,
  IDENTITY: IdentitySubtable<F>,
  LT_ABS: LtAbsSubtable<F>,
  LTU: LtuSubtable<F>,
  OR: OrSubtable<F>,
  SLL0: SllSubtable<F, 0, WORD_SIZE>,
  SLL1: SllSubtable<F, 1, WORD_SIZE>,
  SLL2: SllSubtable<F, 2, WORD_SIZE>,
  SLL3: SllSubtable<F, 3, WORD_SIZE>,
  SRA_SIGN: SraSignSubtable<F, WORD_SIZE>,
  SRL0: SrlSubtable<F, 0, WORD_SIZE>,
  SRL1: SrlSubtable<F, 1, WORD_SIZE>,
  SRL2: SrlSubtable<F, 2, WORD_SIZE>,
  SRL3: SrlSubtable<F, 3, WORD_SIZE>,
  TRUNCATE: TruncateOverflowSubtable<F, WORD_SIZE>,
  XOR: XorSubtable<F>,
  ZERO_LSB: ZeroLSBSubtable<F>
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

  use crate::jolt::instruction::{
    add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
    bgeu::BGEUInstruction, blt::BLTInstruction, bltu::BLTUInstruction, bne::BNEInstruction,
    jal::JALInstruction, jalr::JALRInstruction, or::ORInstruction, sll::SLLInstruction,
    slt::SLTInstruction, sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction,
    sub::SUBInstruction, xor::XORInstruction, JoltInstruction, Opcode,
  };
  use crate::jolt::vm::instruction_lookups::InstructionLookupsProof;
  use crate::{
    jolt::vm::rv32i_vm::{InstructionLookups, Jolt, RV32IJoltVM, C, M, RV32I},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{index_to_field_bitvector, math::Math, random::RandomTape, split_bits},
  };
  use std::any::TypeId;
  use strum::{EnumCount, IntoEnumIterator};
  use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

  fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
    let mut rng = test_rng();
    let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
    for _ in 0..memory_bits {
      r_i.push(F::rand(&mut rng));
    }
    r_i
  }

  #[test]
  fn instruction_lookups() {
    let mut rng = test_rng();

    let ops: Vec<RV32I> = vec![
      RV32I::ADD(ADDInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::AND(ANDInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::BEQ(BEQInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::BGE(BGEInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::BGEU(BGEUInstruction(
        rng.next_u32() as u64,
        rng.next_u32() as u64,
      )),
      RV32I::BLT(BLTInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::BLTU(BLTUInstruction(
        rng.next_u32() as u64,
        rng.next_u32() as u64,
      )),
      RV32I::BNE(BNEInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::JAL(JALInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::JALR(JALRInstruction(
        rng.next_u32() as u64,
        rng.next_u32() as u64,
      )),
      RV32I::OR(ORInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::SLL(SLLInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::SRA(SRAInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::SRL(SRLInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::SUB(SUBInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
      RV32I::XOR(XORInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
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
}
