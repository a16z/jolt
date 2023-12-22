use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::Jolt;
use crate::jolt::instruction::add::ADD32Instruction;
use crate::jolt::instruction::{
    and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction, bgeu::BGEUInstruction,
    blt::BLTInstruction, bltu::BLTUInstruction, bne::BNEInstruction, jal::JALInstruction,
    jalr::JALRInstruction, or::ORInstruction, sll::SLLInstruction, slt::SLTInstruction,
    sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction, sub::SUBInstruction,
    xor::XORInstruction, JoltInstruction, Opcode,
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
        impl $enum_name {
            pub fn random_instruction(rng: &mut StdRng) -> Self {
                let index = rng.next_u64() as usize % $enum_name::COUNT;
                let instruction = $enum_name::iter()
                    .enumerate()
                    .filter(|(i, _)| *i == index)
                    .map(|(_, x)| x)
                    .next()
                    .unwrap();
                instruction.random(rng)
            }
        }
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
    use itertools::Itertools;
    use merlin::Transcript;
    use rand_chacha::rand_core::RngCore;
    use rand_core::SeedableRng;
    use std::collections::HashSet;

    use crate::jolt::instruction;
    use crate::jolt::instruction::{
        add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
        bgeu::BGEUInstruction, blt::BLTInstruction, bltu::BLTUInstruction, bne::BNEInstruction,
        jal::JALInstruction, jalr::JALRInstruction, or::ORInstruction, sll::SLLInstruction,
        slt::SLTInstruction, sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction,
        sub::SUBInstruction, xor::XORInstruction, JoltInstruction, Opcode,
    };
    use crate::jolt::trace::{rv::RVTraceRow, JoltProvableTrace};
    use crate::jolt::vm::bytecode::{BytecodeProof, ELFRow};
    use crate::jolt::vm::instruction_lookups::InstructionLookupsProof;
    use crate::jolt::vm::read_write_memory::ReadWriteMemoryProof;
    use crate::jolt::vm::{JoltProof, MemoryOp};
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
    fn fib_e2e() {
        use common::{path::JoltPaths, serializable::Serializable, ELFInstruction};

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("fibonacci");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let mut bytecode_rows = bytecode.iter().map(ELFRow::from).collect();

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
            .collect();

        let instructions: Vec<RV32I> = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| row.to_jolt_instructions())
            .collect();

        // Emulator sets register 0xb to 0x1020 upon initialization for some reason,
        // something about Linux boot requiring it...
        let mut memory_trace: Vec<MemoryOp> = vec![MemoryOp::Write(11, 4128)];
        memory_trace.extend(converted_trace.into_iter().flat_map(|row| row.to_ram_ops()));
        let next_power_of_two = memory_trace.len().next_power_of_two();
        memory_trace.resize(next_power_of_two, MemoryOp::no_op());

        let mut transcript = Transcript::new(b"Jolt transcript");
        let mut random_tape: RandomTape<EdwardsProjective> =
            RandomTape::new(b"Jolt prover randomness");
        let bytecode_proof: BytecodeProof<Fr, EdwardsProjective> = RV32IJoltVM::prove_bytecode(
            bytecode_rows,
            bytecode_trace,
            &mut transcript,
            &mut random_tape,
        );
        let memory_proof: ReadWriteMemoryProof<Fr, EdwardsProjective> =
            RV32IJoltVM::prove_memory(bytecode, memory_trace, &mut transcript, &mut random_tape);
        let instruction_lookups: InstructionLookupsProof<_, _> =
            RV32IJoltVM::prove_instruction_lookups(instructions, &mut transcript, &mut random_tape);

        let jolt_proof: JoltProof<Fr, EdwardsProjective> = JoltProof {
            instruction_lookups,
            read_write_memory: memory_proof,
            bytecode: bytecode_proof,
        };

        let mut transcript = Transcript::new(b"Jolt transcript");
        assert!(RV32IJoltVM::verify_bytecode(jolt_proof.bytecode, &mut transcript).is_ok());
        assert!(RV32IJoltVM::verify_memory(jolt_proof.read_write_memory, &mut transcript).is_ok());
        assert!(RV32IJoltVM::verify_instruction_lookups(
            jolt_proof.instruction_lookups,
            &mut transcript
        )
        .is_ok());
    }

    #[test]
    fn fib_r1cs() {
        use common::{path::JoltPaths, serializable::Serializable, ELFInstruction};

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("fibonacci");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let mut bytecode_rows: Vec<ELFRow> = bytecode.clone().iter().map(ELFRow::from).collect();

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
            .collect();

        let instructions: Vec<RV32I> = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| row.to_jolt_instructions())
            .collect();

        let instructions_r1cs: Vec<RV32I> = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| {
                let instructions = row.to_jolt_instructions();
                if instructions.is_empty() {
                    vec![ADDInstruction::<32>(0_u64, 0_u64).into()] 
                } else {
                    instructions
                }
            })
            .collect();
    
        let memory_trace_r1cs = converted_trace.clone().into_iter().flat_map(|row| row.to_ram_ops()).collect_vec();

        let circuit_flags = converted_trace.clone()
            .iter()
            .flat_map(|row| {
                let mut flags = row.to_circuit_flags();
                // flags.reverse();
                flags.into_iter() 
            })
            .collect::<Vec<_>>();

        let mut transcript = Transcript::new(b"Jolt transcript");
        let mut random_tape: RandomTape<EdwardsProjective> =
            RandomTape::new(b"Jolt prover randomness");
        RV32IJoltVM::prove_r1cs(
            instructions_r1cs, 
            bytecode_rows,
            bytecode_trace,
            bytecode, 
            memory_trace_r1cs, 
            circuit_flags,
            &mut transcript,
            &mut random_tape,
        );
    }


    #[test]
    fn instruction_lookups() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);
        const NUM_CYCLES: usize = 100;
        let ops: Vec<RV32I> = vec![RV32I::random_instruction(&mut rng); NUM_CYCLES];

        let mut prover_transcript = Transcript::new(b"example");
        let mut random_tape = RandomTape::new(b"test_tape");

        let proof: InstructionLookupsProof<Fr, EdwardsProjective> =
            RV32IJoltVM::prove_instruction_lookups(ops, &mut prover_transcript, &mut random_tape);
        let mut verifier_transcript = Transcript::new(b"example");
        assert!(RV32IJoltVM::verify_instruction_lookups(proof, &mut verifier_transcript).is_ok());
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
}
