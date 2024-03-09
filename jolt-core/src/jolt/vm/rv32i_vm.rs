use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use common::field_conversion::IntoSpartan;
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::Jolt;
use crate::jolt::instruction::add::ADD32Instruction;
use crate::jolt::instruction::{
    and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction, bgeu::BGEUInstruction,
    blt::BLTInstruction, bltu::BLTUInstruction, bne::BNEInstruction, or::ORInstruction,
    sll::SLLInstruction, slt::SLTInstruction, sltu::SLTUInstruction, sra::SRAInstruction,
    srl::SRLInstruction, sub::SUBInstruction, xor::XORInstruction, JoltInstruction, Opcode,
    SubtableIndices,
};
use crate::jolt::subtable::{
    and::AndSubtable, eq::EqSubtable, eq_abs::EqAbsSubtable, eq_msb::EqMSBSubtable,
    gt_msb::GtMSBSubtable, identity::IdentitySubtable, lt_abs::LtAbsSubtable, ltu::LtuSubtable,
    or::OrSubtable, sll::SllSubtable, sra_sign::SraSignSubtable, srl::SrlSubtable,
    truncate_overflow::TruncateOverflowSubtable, xor::XorSubtable, LassoSubtable, SubtableId,
};

/// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
/// are callable on the enum type via enum_dispatch.
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(u8)]
        #[derive(Copy, Clone, Debug, EnumIter, EnumCountMacro)]
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
        impl<F: PrimeField> From<SubtableId> for $enum_name<F> {
          fn from(subtable_id: SubtableId) -> Self {
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
  XOR: XorSubtable<F>
);

// ==================== JOLT ====================

pub enum RV32IJoltVM {}

pub const C: usize = 4;
pub const M: usize = 1 << 16;

impl<F, G> Jolt<'_, F, G, C, M> for RV32IJoltVM
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    G::Affine: IntoSpartan,
{
    type InstructionSet = RV32I;
    type Subtables = RV32ISubtables<F>;
}

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective};
    use common::constants::MEMORY_OPS_PER_INSTRUCTION;
    use common::{path::JoltPaths, serializable::Serializable, ELFInstruction};
    use itertools::Itertools;
    use merlin::Transcript;
    use rand_core::SeedableRng;
    use std::collections::HashSet;

    use crate::jolt::instruction::{add::ADDInstruction, JoltInstruction};
    use crate::jolt::trace::{rv::RVTraceRow, JoltProvableTrace};
    use crate::jolt::vm::bytecode::ELFRow;
    use crate::jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM, C, M, RV32I};
    use crate::jolt::vm::MemoryOp;
    use std::sync::Mutex;
    use strum::{EnumCount, IntoEnumIterator};

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    lazy_static::lazy_static! {
        static ref FIB_FILE_LOCK: Mutex<()> = Mutex::new(());
        static ref SHA3_FILE_LOCK: Mutex<()> = Mutex::new(());
    }

    #[test]
    fn instruction_lookups() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);
        const NUM_CYCLES: usize = 100;
        let ops: Vec<RV32I> = std::iter::repeat_with(|| RV32I::random_instruction(&mut rng))
            .take(NUM_CYCLES)
            .collect();

        let mut prover_transcript = Transcript::new(b"example");

        let preprocessing = RV32IJoltVM::preprocess(1 << 20, 1 << 20, 1 << 22);

        let (proof, _, commitment) =
            <RV32IJoltVM as Jolt<'_, _, G1Projective, C, M>>::prove_instruction_lookups(
                &preprocessing.instruction_lookups,
                ops,
                &preprocessing.generators,
                &mut prover_transcript,
            );
        let mut verifier_transcript = Transcript::new(b"example");
        assert!(RV32IJoltVM::verify_instruction_lookups(
            &preprocessing.instruction_lookups,
            proof,
            commitment,
            &mut verifier_transcript
        )
        .is_ok());
    }

    #[test]
    fn instruction_set_subtables() {
        let mut subtable_set: HashSet<_> = HashSet::new();
        for instruction in <RV32IJoltVM as Jolt<_, G1Projective, C, M>>::InstructionSet::iter() {
            for (subtable, _) in instruction.subtables::<Fr>(C, M) {
                // panics if subtable cannot be cast to enum variant
                let _ = <RV32IJoltVM as Jolt<_, G1Projective, C, M>>::Subtables::from(
                    subtable.subtable_id(),
                );
                subtable_set.insert(subtable.subtable_id());
            }
        }
        assert_eq!(
            subtable_set.len(),
            <RV32IJoltVM as Jolt<_, G1Projective, C, M>>::Subtables::COUNT,
            "Unused enum variants in Subtables"
        );
    }

    #[test]
    fn fib_r1cs() {
        let _guard = FIB_FILE_LOCK.lock().unwrap();
        compiler::cached_compile_example("fibonacci");

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("fibonacci");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let bytecode_rows: Vec<ELFRow> = bytecode.clone().iter().map(ELFRow::from).collect();

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
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

        let memory_trace_r1cs = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| row.to_ram_ops())
            .collect_vec();

        let circuit_flags = converted_trace
            .clone()
            .iter()
            .flat_map(|row| row.to_circuit_flags::<Fr>())
            .collect::<Vec<_>>();

        let mut transcript = Transcript::new(b"Jolt transcript");

        let preprocessing = RV32IJoltVM::preprocess(1 << 20, 1 << 20, 1 << 20);
        <RV32IJoltVM as Jolt<'_, _, G1Projective, C, M>>::prove_r1cs(
            preprocessing, 
            instructions_r1cs,
            bytecode_rows,
            bytecode_trace,
            bytecode,
            memory_trace_r1cs,
            circuit_flags,
            &mut transcript,
        );
    }

    #[test]
    fn fib_e2e() {
        use common::{path::JoltPaths, serializable::Serializable, ELFInstruction};
        let _guard = FIB_FILE_LOCK.lock().unwrap();
        compiler::cached_compile_example("fibonacci");

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("fibonacci");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
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

        let memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> = converted_trace
            .clone()
            .into_iter()
            .map(|row| row.to_ram_ops().try_into().unwrap())
            .collect();
        let circuit_flags = converted_trace
            .iter()
            .flat_map(|row| row.to_circuit_flags::<Fr>())
            .collect::<Vec<_>>();

        let preprocessing = RV32IJoltVM::preprocess(1 << 20, 1 << 20, 1 << 20);
        let (proof, commitments) = <RV32IJoltVM as Jolt<Fr, G1Projective, C, M>>::prove(
            bytecode,
            bytecode_trace,
            memory_trace,
            instructions_r1cs,
            circuit_flags,
            preprocessing.clone(),
        );
        let verification_result = RV32IJoltVM::verify(preprocessing, proof, commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn sha3_r1cs() {
        use common::{path::JoltPaths, serializable::Serializable, ELFInstruction};
        compiler::cached_compile_example("sha3-ex");

        let trace_location = JoltPaths::trace_path("sha3-ex");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("sha3-ex");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let bytecode_rows: Vec<ELFRow> = bytecode.clone().iter().map(ELFRow::from).collect();

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
            .collect();

        // R1CS expects a single lookup instruction per
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

        let memory_trace_r1cs = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| row.to_ram_ops())
            .collect_vec();

        let circuit_flags = converted_trace
            .clone()
            .iter()
            .flat_map(|row| row.to_circuit_flags())
            .collect::<Vec<_>>();

        let mut transcript = Transcript::new(b"Jolt transcript");
        let preprocessing = RV32IJoltVM::preprocess(1 << 20, 1 << 20, 1 << 20);
        <RV32IJoltVM as Jolt<'_, _, G1Projective, C, M>>::prove_r1cs(
            preprocessing, 
            instructions_r1cs,
            bytecode_rows,
            bytecode_trace,
            bytecode,
            memory_trace_r1cs,
            circuit_flags,
            &mut transcript,
        );
    }

    #[test]
    fn sha3_e2e() {
        let _guard = SHA3_FILE_LOCK.lock().unwrap();
        compiler::cached_compile_example("sha3-ex");

        let trace_location = JoltPaths::trace_path("sha3-ex");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("sha3-ex");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
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

        let memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> = converted_trace
            .clone()
            .into_iter()
            .map(|row| row.to_ram_ops().try_into().unwrap())
            .collect_vec();

        let circuit_flags = converted_trace
            .clone()
            .iter()
            .flat_map(|row| row.to_circuit_flags::<Fr>())
            .collect::<Vec<_>>();

        let preprocessing = RV32IJoltVM::preprocess(1 << 20, 1 << 20, 1 << 20);
        let (jolt_proof, jolt_commitments) =
            <RV32IJoltVM as Jolt<'_, _, G1Projective, C, M>>::prove(
                bytecode,
                bytecode_trace,
                memory_trace,
                instructions_r1cs,
                circuit_flags,
                preprocessing.clone(),
            );

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
