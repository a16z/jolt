use crate::backend::trace::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_claims::protocols::jolt::JoltOneHotConfig;
use jolt_program::{
    execution::{JoltProgram, TraceData, TraceOutput},
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
use jolt_riscv::instructions::VirtualAdvice;
use jolt_riscv::{
    CircuitFlags, Flags, JoltInstruction, JoltInstructionKind, JoltInstructionRow,
    JoltTraceRow as TraceRow, NormalizedOperands, RamAccess, RamRead, RamWrite, RegisterRead,
    RegisterState, RegisterWrite, RV64IMAC_JOLT,
};
use std::sync::Arc;

pub fn supported_jolt_kinds() -> Vec<JoltInstructionKind> {
    JoltInstructionKind::ALL
        .iter()
        .copied()
        .filter(|&kind| RV64IMAC_JOLT.jolt_dense_index(kind).is_some())
        .collect()
}

fn all_kinds_operands(kind: JoltInstructionKind, slot: usize) -> NormalizedOperands {
    let magnitude = 13 + 5 * slot as i128;
    let imm = if slot.is_multiple_of(2) {
        -magnitude
    } else {
        magnitude
    };
    let flags = JoltInstruction::from(JoltInstructionRow {
        instruction_kind: kind,
        ..JoltInstructionRow::default()
    })
    .circuit_flags();
    let (rd, rs1, rs2) = match kind {
        _ if flags[CircuitFlags::Store] => (None, Some(2), Some(3)),
        _ if flags[CircuitFlags::Load] => (Some(1), Some(2), None),
        JoltInstructionKind::BEQ | JoltInstructionKind::VirtualAssertEQ => (None, Some(2), Some(3)),
        JoltInstructionKind::ADDI
        | JoltInstructionKind::VirtualMULI
        | JoltInstructionKind::VirtualMovsign => (Some(1), Some(2), None),
        JoltInstructionKind::LUI | JoltInstructionKind::JAL => (Some(1), None, None),
        _ if kind == JoltInstructionKind::VirtualAdvice(VirtualAdvice(())) => (Some(1), None, None),
        _ => (Some(1), Some(2), Some(3)),
    };
    NormalizedOperands { rd, rs1, rs2, imm }
}

#[expect(clippy::expect_used, reason = "test fixture construction")]
pub fn all_kinds_backend(seed: u64) -> (TraceBackend, usize, usize) {
    let kinds = supported_jolt_kinds();
    let bytecode: Vec<JoltInstructionRow> = kinds
        .iter()
        .enumerate()
        .map(|(slot, &instruction_kind)| JoltInstructionRow {
            instruction_kind,
            address: 0x8000_0000 + 4 * slot,
            operands: all_kinds_operands(instruction_kind, slot),
            virtual_sequence_remaining: (instruction_kind != JoltInstructionKind::NoOp
                && slot % 2 == 0)
                .then_some(0),
            is_first_in_sequence: instruction_kind != JoltInstructionKind::NoOp && slot % 4 == 1,
            is_compressed: instruction_kind != JoltInstructionKind::NoOp && slot % 5 == 4,
        })
        .collect();

    let log_t = kinds.len().next_power_of_two().ilog2() as usize;
    let cycles = 1usize << log_t;
    let ram_k = 1usize << 6;
    let entry_address = bytecode[0].address as u64;
    let memory_layout = MemoryLayout::new(&MemoryConfig {
        program_size: Some(1 << 12),
        ..MemoryConfig::default()
    });
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(bytecode.clone(), entry_address, RV64IMAC_JOLT)
            .expect("all-kinds bytecode fixture"),
        ram: RAMPreprocessing::default(),
        memory_layout: memory_layout.clone(),
        max_padded_trace_length: cycles,
    });

    let lowest = memory_layout.get_lowest_address();
    let jumping_slot = bytecode
        .iter()
        .position(|&row| JoltInstruction::from(row).circuit_flags()[CircuitFlags::Jump]);
    let rows: Vec<TraceRow> = (0..cycles)
        .map(|cycle| {
            let slot = match jumping_slot {
                Some(slot) if cycle + 1 == cycles => slot,
                _ => cycle % bytecode.len(),
            };
            let instruction = bytecode[slot];
            if instruction.instruction_kind == JoltInstructionKind::NoOp {
                return TraceRow::default();
            }
            let mix = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(cycle as u64);
            let decoded = JoltInstruction::from(instruction);
            let flags = decoded.circuit_flags();
            let post_value = mix.rotate_left(29);
            let rs2_value = mix.rotate_left(17) | 1;
            let ram_access = if flags[CircuitFlags::Load] {
                RamAccess::Read(RamRead {
                    address: lowest + 8 * ((cycle % ram_k) as u64),
                    value: post_value,
                })
            } else if flags[CircuitFlags::Store] {
                RamAccess::Write(RamWrite {
                    address: lowest + 8 * ((cycle % ram_k) as u64),
                    pre_value: mix.rotate_left(9),
                    post_value: rs2_value,
                })
            } else {
                RamAccess::NoOp
            };
            TraceRow::new(
                instruction,
                RegisterState {
                    rs1: instruction.operands.rs1.map(|register| RegisterRead {
                        register,
                        value: mix | 1,
                    }),
                    rs2: instruction
                        .operands
                        .rs2
                        .filter(|_| !flags[CircuitFlags::Load])
                        .map(|register| RegisterRead {
                            register,
                            value: rs2_value,
                        }),
                    rd: instruction
                        .operands
                        .rd
                        .filter(|_| !flags[CircuitFlags::Store])
                        .map(|register| RegisterWrite {
                            register,
                            pre_value: mix.rotate_left(23),
                            post_value,
                        }),
                },
                ram_access,
                preprocessing
                    .bytecode
                    .get_pc(&instruction)
                    .expect("fixture PC") as u32,
            )
            .expect("checked fixture row")
        })
        .collect();

    let program = Arc::new(JoltProgram::default());
    let trace = TraceOutput::new(TraceData::new(rows), Default::default(), None, None);
    let backend = TraceBackend::new(
        JoltVmWitnessConfig::new(
            log_t,
            ram_k,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        ),
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );
    (backend, cycles, kinds.len())
}
