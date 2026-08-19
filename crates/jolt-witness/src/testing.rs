//! Sample-trace fixtures for the derive-generated bundle consistency tests.

use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltPolynomialId};
use jolt_field::Fr;
use jolt_program::{
    execution::{
        JoltProgram, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead, RegisterState,
        RegisterWrite, TraceOutput, TraceRow,
    },
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
use jolt_riscv::{
    CircuitFlags, Flags, JoltInstruction, JoltInstructionKind, JoltInstructionRow,
    NormalizedOperands, RV64IMAC_JOLT,
};
use std::sync::Arc;

use crate::backend::trace::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use crate::{BundleSource, JoltWitnessOracle, WitnessBundle};

/// Runs `f` against a small canned backend: two real cycles (an ADDI with
/// register activity and RAM traffic, then a RAM write) padded to `2^2`.
pub fn with_sample_backend<R>(f: impl FnOnce(&TraceBackend<OwnedTrace>) -> R) -> R {
    with_sample_backend_config(64, 0x8000_1000, f)
}

#[expect(clippy::unwrap_used, reason = "test fixture construction")]
fn with_sample_backend_config<R>(
    ram_k: usize,
    ram_base: u64,
    f: impl FnOnce(&TraceBackend<OwnedTrace>) -> R,
) -> R {
    let instruction = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::ADDI,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: 3,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    };
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(
            vec![instruction],
            instruction.address as u64,
            RV64IMAC_JOLT,
        )
        .unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 4,
    });
    let program = Arc::new(JoltProgram::default());
    let rows = vec![
        TraceRow {
            instruction,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 5,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 0,
                    post_value: 8,
                }),
                ..Default::default()
            },
            ram_access: RamAccess::Read(RamRead {
                address: ram_base,
                value: 7,
            }),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        },
        TraceRow {
            ram_access: RamAccess::Write(RamWrite {
                address: ram_base + 8,
                pre_value: 7,
                post_value: 11,
            }),
            ..Default::default()
        },
    ];
    let config = JoltVmWitnessConfig::new(
        2,
        ram_k,
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    );
    let inputs = JoltVmWitnessInputs::new(
        &program,
        &preprocessing,
        TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
    );
    let backend = TraceBackend::new(config, inputs);
    f(&backend)
}

pub fn with_ram_sized_backend<R>(f: impl FnOnce(&TraceBackend<OwnedTrace>) -> R) -> R {
    with_sample_backend_config(RAM_SIZED_K, RAM_SIZED_BASE, f)
}

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
    let (rd, rs1, rs2) = match kind {
        JoltInstructionKind::SD
        | JoltInstructionKind::BEQ
        | JoltInstructionKind::VirtualAssertEQ => (None, Some(2), Some(3)),
        JoltInstructionKind::LD
        | JoltInstructionKind::ADDI
        | JoltInstructionKind::VirtualMULI
        | JoltInstructionKind::VirtualMovsign => (Some(1), Some(2), None),
        JoltInstructionKind::LUI | JoltInstructionKind::JAL => (Some(1), None, None),
        _ if kind
            == JoltInstructionKind::VirtualAdvice(jolt_riscv::instructions::VirtualAdvice(())) =>
        {
            (Some(1), None, None)
        }
        _ => (Some(1), Some(2), Some(3)),
    };
    NormalizedOperands { rd, rs1, rs2, imm }
}

#[expect(clippy::expect_used, reason = "test fixture construction")]
pub fn all_kinds_backend(seed: u64) -> (TraceBackend<OwnedTrace>, usize, usize) {
    let kinds = supported_jolt_kinds();
    let bytecode: Vec<JoltInstructionRow> = kinds
        .iter()
        .enumerate()
        .map(|(slot, &instruction_kind)| JoltInstructionRow {
            instruction_kind,
            address: 0x8000_0000 + 4 * slot,
            operands: all_kinds_operands(instruction_kind, slot),
            virtual_sequence_remaining: match slot % 3 {
                0 => None,
                1 => Some(0),
                _ => Some((slot % 5 + 1) as u16),
            },
            is_first_in_sequence: slot % 4 == 1,
            is_compressed: slot % 5 == 4,
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
    let jumping_slot = bytecode.iter().position(|&row| {
        JoltInstruction::try_from(row)
            .is_ok_and(|decoded| decoded.circuit_flags()[CircuitFlags::Jump])
    });
    let rows: Vec<TraceRow> = (0..cycles)
        .map(|cycle| {
            let slot = match jumping_slot {
                Some(slot) if cycle + 1 == cycles => slot,
                _ => cycle % bytecode.len(),
            };
            let instruction = bytecode[slot];
            let mix = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(cycle as u64);
            let ram_access = match cycle % 3 {
                0 => RamAccess::NoOp,
                1 => RamAccess::Read(RamRead {
                    address: lowest + 8 * ((cycle % ram_k) as u64),
                    value: mix.rotate_left(5),
                }),
                _ => RamAccess::Write(RamWrite {
                    address: lowest + 8 * ((cycle % ram_k) as u64),
                    pre_value: mix.rotate_left(9),
                    post_value: mix.rotate_left(13),
                }),
            };
            TraceRow {
                instruction,
                registers: RegisterState {
                    rs1: instruction.operands.rs1.map(|register| RegisterRead {
                        register,
                        value: mix | 1,
                    }),
                    rs2: instruction.operands.rs2.map(|register| RegisterRead {
                        register,
                        value: mix.rotate_left(17) | 1,
                    }),
                    rd: instruction.operands.rd.map(|register| RegisterWrite {
                        register,
                        pre_value: mix.rotate_left(23),
                        post_value: mix.rotate_left(29),
                    }),
                },
                #[cfg(feature = "field-inline")]
                field_inline: None,
                ram_access,
            }
        })
        .collect();

    let program = Arc::new(JoltProgram::default());
    let trace = TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None);
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

const RAM_SIZED_K: usize = 16;

const RAM_SIZED_BASE: u64 = 0x40;

/// Asserts that one annotated bundle field's column (extracted by `value`)
/// equals the backend's `oracle_table` for `id` — the typed path and the id
/// path meeting at the `Extract` impls. Driven by the derive-generated
/// per-field consistency tests.
#[expect(clippy::unwrap_used, reason = "test assertion helper")]
pub fn assert_bundle_column_matches<B>(id: JoltPolynomialId, value: impl Fn(&B) -> Fr)
where
    B: WitnessBundle + Copy + Send + Sync,
{
    with_sample_backend(|backend| {
        assert!(
            B::annotated_ids().contains(&id),
            "{id:?} is not in the bundle's annotated id set"
        );
        let rows: Vec<B> = backend.bundles().unwrap();
        let column: Vec<Fr> = rows.iter().map(value).collect();
        let table = JoltWitnessOracle::<Fr>::oracle_table(backend, id).unwrap();
        assert_eq!(
            column, table,
            "bundle column diverges from oracle_table for {id:?}"
        );
    });
}
