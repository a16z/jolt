use ark_ec::CurveGroup;
use ark_ff::PrimeField;
// use common::RVTraceRow;
use enum_dispatch::enum_dispatch;
use pasta_curves::group::Curve;
use pasta_curves::pallas::Scalar;
use std::any::TypeId;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};
use merlin::Transcript;
use ark_std::test_rng;

use super::Jolt;
use super::read_write_memory::{MemoryCommitment, MemoryOp, ReadWriteMemory};
use super::pc::{ELFRow, FiveTuplePoly, PCPolys};
use crate::jolt::trace::JoltProvableTrace;
use crate::jolt::trace::rv::RVTraceRow;
use crate::jolt::instruction::add::ADD32Instruction;
use crate::jolt::instruction::{
    and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
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

/// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
/// are callable on the enum type via enum_dispatch.
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(u8)]
        #[derive(Copy, Clone, EnumIter, EnumCountMacro)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name { $($alias($struct)),+ }
        impl Opcode for $enum_name {}
    };
}

// TODO(moodlezoup): Consider replacing From<TypeId> and Into<usize> with
//     combined trait/function to_enum_index(subtable: &dyn LassoSubtable<F>) => usize
/// Generates an enum out of a list of LassoSubtable types. All LassoSubtable methods
/// are callable on the enum type via enum_dispatch.
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
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
  SLT: SLTInstruction,
  SLTU: SLTUInstruction,
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

impl<F, G> Jolt<F, G, C, M> for RV32IJoltVM
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type InstructionSet = RV32I;
    type Subtables = RV32ISubtables<F>;
}

pub struct RV32IJoltVM2<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    memory_size: usize, 
    program: Vec<ELFRow>, 
    trace: Vec<RVTraceRow>,
    _p1: std::marker::PhantomData<F>,
    _p2: std::marker::PhantomData<G>,
}

const C: usize = 4;
const M: usize = 1 << 16;

impl<F, G> Jolt<F, G, C, M> for RV32IJoltVM2<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type InstructionSet = RV32I;
    type Subtables = RV32ISubtables<F>;

    fn prove(&self) {
        let mut transcript = Transcript::new(b"jolt");

        // Prepare Lookups 
        let instruction_ops =   
            self
            .trace
            .iter()
            .flat_map(|row| row.to_jolt_instructions())
            .collect::<Vec<_>>();

        // let r: Vec<F> = (0..instruction_ops.len()).map(|_| F::one()).collect();
        let mut rng = test_rng();
        let mut r: Vec<F> = Vec::with_capacity(instruction_ops.len());
        for _ in 0..instruction_ops.len() {
            r.push(F::rand(&mut rng));
        }



        // // Prepare program code
        // let elf_rows = 
        //     self
        //     .trace
        //     .iter()
        //     .map(|row| row.to_pc_trace())
        //     .collect::<Vec<_>>();

        // let pc_polys = PCPolys::<F, G>::new_program(
        //     self.program.clone(), 
        //     elf_rows
        // );

        // let circuit_flags = 
        //     self
        //     .trace
        //     .iter()
        //     .flat_map(|row| row.to_circuit_flags::<F>())
        //     .collect::<Vec<_>>();

        // // Prove RAM and Registers  
        // let memory_trace = 
        //     self
        //     .trace
        //     .iter()
        //     .flat_map(|row| row.to_ram_ops())
        //     .collect::<Vec<_>>();

        // let (rw_memory, _) = ReadWriteMemory::<F, G>::new(
        //     memory_trace.clone(),
        //     self.memory_size, 
        //     &mut transcript,
        // );

        Self::prove_instruction_lookups(instruction_ops.clone(), r, &mut transcript);
        // // TODO: call pc prover 
        // Self::prove_memory(memory_trace, self.memory_size, &mut transcript);
        // Self::prove_r1cs(
        //     self.trace.len(),
        //     pc_polys,
        //     rw_memory,
        //     instruction_ops,
        //     circuit_flags,
        // );
    }
}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use super::RV32IJoltVM2; 
    use ark_curve25519::{EdwardsProjective, Fr};
    use ark_ec::CurveGroup;
    use ark_ff::PrimeField;
    use ark_std::{log2, test_rng, One, Zero};
    use enum_dispatch::enum_dispatch;
    use merlin::Transcript;
    use rand_chacha::rand_core::RngCore;
    use std::collections::HashSet;

    use common::{constants::REGISTER_COUNT, RV32InstructionFormat, RV32IM};
    use crate::jolt::vm::pc::{ELFRow, FiveTuplePoly, PCPolys};
    use crate::jolt::trace::rv::RVTraceRow;
    use crate::jolt::instruction::{
        add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
        bgeu::BGEUInstruction, blt::BLTInstruction, bltu::BLTUInstruction, bne::BNEInstruction,
        jal::JALInstruction, jalr::JALRInstruction, or::ORInstruction, sll::SLLInstruction,
        slt::SLTInstruction, sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction,
        sub::SUBInstruction, xor::XORInstruction, JoltInstruction, Opcode,
    };
    use crate::jolt::vm::instruction_lookups::InstructionLookupsProof;
    use crate::{
        jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM, C, M, RV32I},
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
        assert!(
            RV32IJoltVM::verify_instruction_lookups(proof, r, &mut verifier_transcript).is_ok()
        );
    }

    #[test]
    fn instruction_set_subtables() {
        let mut subtable_set: HashSet<_> = HashSet::new();
        for instruction in <RV32IJoltVM as Jolt<_, EdwardsProjective, C, M>>::InstructionSet::iter()
        {
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

    #[test]
    fn rv32ijoltvm_tiny_trace() {
        let program = vec![
            ELFRow::new(0, 2u64, 2u64, 2u64, 2u64, 2u64),
            ELFRow::new(1, 4u64, 4u64, 4u64, 4u64, 4u64),
            ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
        ];
        let trace = vec![
            // ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
            // ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
            RVTraceRow::new(1, RV32IM::AND, Some(4u64), Some(4u64), Some(4u64), Some(4u32), Some(0), Some(0), Some(0), Some(0), None, None),
            // RVTraceRow::new(1, RV32IM::SUB, Some(2u64), Some(2u64), Some(2u64), Some(2u32), Some(0), Some(0), Some(0), Some(0), None, None),
        ];

        let mut vm = RV32IJoltVM2::<Fr, EdwardsProjective> {
            memory_size: 1024,
            program,
            trace,
            _p1: std::marker::PhantomData,
            _p2: std::marker::PhantomData,
        };

        vm.prove();
    }
}
