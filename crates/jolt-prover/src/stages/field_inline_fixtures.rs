//! Shared FR-profile trace fixtures for the stage-recipe round-trip tests.
//!
//! Hand-crafted rows that are semantically consistent instruction executions
//! (the same discipline as `jolt_witness::testing::with_sample_backend`), so
//! the composed R1CS eq rows are satisfied and the stage sumchecks' hard
//! self-checks hold. Two profiles: an ADDI-only trace (an FR-profile guest
//! executing zero FR instructions — every FR column is zero), and an FR
//! arithmetic trace (two field loads and a multiply, the stage-0 fixture's
//! rows) whose decoded FR instruction words populate the FR columns.

use std::sync::Arc;

use common::constants::RAM_START_ADDRESS;
use jolt_claims::protocols::jolt::JoltOneHotConfig;
#[cfg(not(feature = "zk"))]
use jolt_program::execution::RegisterRead;
use jolt_program::execution::{
    JoltProgram, OwnedTrace, RegisterState, RegisterWrite, TraceOutput, TraceRow,
};
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{
    FieldInlineOp, JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow,
    NormalizedOperands, RV64IMAC_JOLT_FIELD_INLINE,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

pub(crate) const ENTRY: u64 = RAM_START_ADDRESS;
// 3, not 2: the last physical cycle must be a noop (constraint 21's
// ShouldJump convention), so the FR fixture's four real rows need padding
// room behind them.
pub(crate) const LOG_T: usize = 3;

fn instruction(
    instruction_kind: JoltInstructionKind,
    offset: usize,
    rd: Option<u8>,
    rs1: Option<u8>,
    rs2: Option<u8>,
    imm: i128,
) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind,
        address: ENTRY as usize + offset * 4,
        operands: NormalizedOperands { rd, rs1, rs2, imm },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

#[expect(clippy::unwrap_used, reason = "test fixture construction")]
pub(crate) fn fr_backend(
    bytecode: Vec<JoltInstructionRow>,
    rows: Vec<TraceRow>,
) -> TraceBackend<OwnedTrace> {
    let profile: JoltInstructionProfile = RV64IMAC_JOLT_FIELD_INLINE;
    let program = Arc::new(JoltProgram::from_parts_with_profile(
        Vec::new(),
        bytecode.clone(),
        Vec::new(),
        ENTRY + 4,
        ENTRY,
        profile,
    ));
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, profile).unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: test_memory_layout(),
        max_padded_trace_length: 1 << LOG_T,
    });
    TraceBackend::new(
        JoltVmWitnessConfig::new(
            LOG_T,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        ),
        JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(OwnedTrace::new(rows), test_public_io(), None, None),
        ),
    )
}

fn enc(value: u64) -> FieldEncodedValue {
    FieldEncodedValue::from_u64(value)
}

fn field_row(instruction: JoltInstructionRow, data: FieldInlineTraceData) -> TraceRow {
    TraceRow {
        instruction,
        field_inline: Some(data.into()),
        ..TraceRow::default()
    }
}

/// A terminal JAL row: the only hand-craftable last real instruction — its
/// `Jump` flag turns off the otherwise-unconditional PC-update row 16, and
/// `ShouldJump` stays 0 because the successor is the noop padding — with the
/// link write (`rd = address + 4`) row 13 demands.
fn halt_jal_row(offset: usize, rd: u8) -> TraceRow {
    let jal = instruction(JoltInstructionKind::JAL, offset, Some(rd), None, None, 0);
    TraceRow {
        instruction: jal,
        registers: RegisterState {
            rd: Some(RegisterWrite {
                register: rd,
                pre_value: 0,
                post_value: ENTRY + (offset as u64) * 4 + 4,
            }),
            ..Default::default()
        },
        ..TraceRow::default()
    }
}

/// An FR-profile guest executing only ordinary instructions (an ADDI with
/// consistent register semantics, then the terminal JAL): the rv64 eq rows
/// are satisfied while every FR column is zero.
// The ZK stage tests exercise only the FR-arithmetic profile.
#[cfg(not(feature = "zk"))]
pub(crate) fn addi_only_backend() -> TraceBackend<OwnedTrace> {
    let addi = instruction(JoltInstructionKind::ADDI, 0, Some(1), Some(2), None, 3);
    let jal = halt_jal_row(1, 5);
    let rows = vec![
        TraceRow {
            instruction: addi,
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
            ..TraceRow::default()
        },
        jal.clone(),
    ];
    fr_backend(vec![addi, jal.instruction], rows)
}

/// Two field loads and a multiply: `FieldRdInc = [13, 17, 221, 0]`,
/// `13 · 17 = 221` — every FR eq row and both FR product lanes are satisfied
/// (the product columns are extractor-derived), and the x-register file is
/// untouched.
pub(crate) fn fr_arithmetic_backend() -> TraceBackend<OwnedTrace> {
    let load_a = instruction(
        JoltInstructionKind::FIELD_LOAD_IMM,
        0,
        Some(1),
        None,
        None,
        13,
    );
    let load_b = instruction(
        JoltInstructionKind::FIELD_LOAD_IMM,
        1,
        Some(2),
        None,
        None,
        17,
    );
    let mul = instruction(
        JoltInstructionKind::FIELD_MUL,
        2,
        Some(3),
        Some(1),
        Some(2),
        0,
    );
    let jal = halt_jal_row(3, 5);
    let rows = vec![
        field_row(
            load_a,
            FieldInlineTraceData {
                op: Some(FieldInlineOp::LoadImm),
                rd: Some(FieldRegisterWrite {
                    register: 1,
                    pre_value: enc(0),
                    post_value: enc(13),
                }),
                ..FieldInlineTraceData::default()
            },
        ),
        field_row(
            load_b,
            FieldInlineTraceData {
                op: Some(FieldInlineOp::LoadImm),
                rd: Some(FieldRegisterWrite {
                    register: 2,
                    pre_value: enc(0),
                    post_value: enc(17),
                }),
                ..FieldInlineTraceData::default()
            },
        ),
        field_row(
            mul,
            FieldInlineTraceData {
                op: Some(FieldInlineOp::Mul),
                rs1: Some(FieldRegisterRead {
                    register: 1,
                    value: enc(13),
                }),
                rs2: Some(FieldRegisterRead {
                    register: 2,
                    value: enc(17),
                }),
                rd: Some(FieldRegisterWrite {
                    register: 3,
                    pre_value: enc(0),
                    post_value: enc(221),
                }),
                product: Some(enc(221)),
                ..FieldInlineTraceData::default()
            },
        ),
        jal.clone(),
    ];
    fr_backend(vec![load_a, load_b, mul, jal.instruction], rows)
}

/// The stage recipes' derived-config shape for the fixture traces: the same
/// derivation `ProverConfig::derive` performs, at the fixture's scale (no
/// RAM traffic, so `ram_K` stays at a small power of two).
pub(crate) fn test_prover_config() -> crate::ProverConfig {
    // Matches the witness backend's `JoltVmWitnessConfig` ram size (64).
    const RAM_LOG_K: usize = 6;
    crate::ProverConfig {
        trace_length: 1 << LOG_T,
        ram_K: 1 << RAM_LOG_K,
        rw_config: jolt_claims::protocols::jolt::JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: LOG_T as u8,
            ram_rw_phase2_num_rounds: RAM_LOG_K as u8,
            registers_rw_phase1_num_rounds: LOG_T as u8,
            registers_rw_phase2_num_rounds: common::constants::REGISTER_COUNT.ilog2() as u8,
        },
        one_hot_config: JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
        trace_polynomial_order: Default::default(),
    }
}

/// A well-formed memory layout for the fixture traces (the default layout is
/// degenerate: its lowest mapped address is zero, which `PublicIoMemory`
/// rejects).
pub(crate) fn test_memory_layout() -> common::jolt_device::MemoryLayout {
    common::jolt_device::MemoryLayout::new(&common::jolt_device::MemoryConfig {
        program_size: Some(1024),
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        max_input_size: 8,
        max_output_size: 8,
        stack_size: 8,
        heap_size: 8,
    })
}

/// The fixture traces' program I/O: empty, over [`test_memory_layout`].
pub(crate) fn test_public_io() -> common::jolt_device::JoltDevice {
    common::jolt_device::JoltDevice {
        memory_layout: test_memory_layout(),
        ..Default::default()
    }
}
