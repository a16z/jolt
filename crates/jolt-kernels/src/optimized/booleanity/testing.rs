use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId};
use jolt_field::Fr;
use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
use jolt_riscv::{
    JoltTraceRow as TraceRow, RamAccess, RamRead, RamWrite, RegisterRead, RegisterState,
    RegisterWrite,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend};

const LOAD: JoltInstructionRow = JoltInstructionRow {
    instruction_kind: JoltInstructionKind::LD,
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
const STORE: JoltInstructionRow = JoltInstructionRow {
    instruction_kind: JoltInstructionKind::SD,
    address: 0x8000_0004,
    operands: NormalizedOperands {
        rd: None,
        rs1: Some(1),
        rs2: Some(3),
        imm: 8,
    },
    ..LOAD
};
const ALU: JoltInstructionRow = JoltInstructionRow {
    instruction_kind: JoltInstructionKind::ADDI,
    address: 0x8000_0008,
    operands: NormalizedOperands {
        rd: Some(1),
        rs1: Some(2),
        rs2: None,
        imm: 3,
    },
    ..LOAD
};

pub(crate) fn load_row(address: u64) -> TraceRow {
    TraceRow::new(
        LOAD,
        RegisterState {
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
        RamAccess::Read(RamRead { address, value: 8 }),
        1,
    )
    .unwrap()
}

pub(crate) fn store_row(address: u64) -> TraceRow {
    TraceRow::new(
        STORE,
        RegisterState {
            rs1: Some(RegisterRead {
                register: 1,
                value: 8,
            }),
            rs2: Some(RegisterRead {
                register: 3,
                value: 11,
            }),
            ..Default::default()
        },
        RamAccess::Write(RamWrite {
            address,
            pre_value: 7,
            post_value: 11,
        }),
        2,
    )
    .unwrap()
}

pub(crate) fn no_op_row() -> TraceRow {
    TraceRow::new(
        JoltInstructionRow::default(),
        RegisterState::default(),
        RamAccess::NoOp,
        0,
    )
    .unwrap()
}

/// An ALU row whose lookup index follows its first operand `value`.
pub(crate) fn alu_row(value: u64) -> TraceRow {
    TraceRow::new(
        ALU,
        RegisterState {
            rs1: Some(RegisterRead { register: 2, value }),
            rd: Some(RegisterWrite {
                register: 1,
                pre_value: 8,
                post_value: 11,
            }),
            ..Default::default()
        },
        RamAccess::NoOp,
        3,
    )
    .unwrap()
}

/// `len` rows of the fixture program: loads and stores over eleven RAM
/// addresses, ALU rows of varied operands, and no-ops.
pub(crate) fn mixed_rows(len: usize) -> Vec<TraceRow> {
    (0..len as u64)
        .map(|cycle| {
            let address = 0x8000_1000 + 8 * (cycle % 11);
            match cycle % 5 {
                0 => load_row(address),
                1 => alu_row(cycle.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                2 => store_row(address),
                3 => no_op_row(),
                _ => alu_row(cycle),
            }
        })
        .collect()
}

pub(crate) fn with_booleanity_backend<R>(
    log_t: usize,
    log_k_chunk: u8,
    f: impl FnOnce(&TraceBackend, BooleanityDimensions) -> R,
) -> R {
    let mut rows = vec![
        load_row(0x8000_1000),
        store_row(0x8000_1008),
        no_op_row(),
        alu_row(5),
    ];
    rows.truncate(1 << log_t);
    with_trace_backend(log_t, log_k_chunk, rows, f)
}

/// Runs `f` against a trace backend over `rows` of the fixture program
/// (padded by the backend up to `2^log_t`). Booleanity dimensions are
/// probed off the backend's own servable set so test and backend
/// geometry cannot drift.
pub(crate) fn with_trace_backend<R>(
    log_t: usize,
    log_k_chunk: u8,
    rows: Vec<TraceRow>,
    f: impl FnOnce(&TraceBackend, BooleanityDimensions) -> R,
) -> R {
    use std::sync::Arc;
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(
            vec![LOAD, STORE, ALU],
            LOAD.address as u64,
            RV64IMAC_JOLT,
        )
        .unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 4.max(1 << log_t),
    });
    let program = Arc::new(JoltProgram::default());

    let config = JoltVmWitnessConfig::new(
        log_t,
        64,
        JoltOneHotConfig {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    );
    let inputs = JoltVmWitnessInputs::new(
        &program,
        &preprocessing,
        TraceOutput::new(TraceData::new(rows), Default::default(), None, None),
    );
    let backend = TraceBackend::new(config, inputs);

    let probe = |family: fn(usize) -> JoltCommittedPolynomial| {
        (0..64)
            .take_while(|index| {
                JoltWitnessOracle::<Fr>::shape(
                    &backend,
                    JoltPolynomialId::Committed(family(*index)),
                )
                .is_ok()
            })
            .count()
    };
    let layout = JoltRaPolynomialLayout::new(
        probe(JoltCommittedPolynomial::InstructionRa),
        probe(JoltCommittedPolynomial::BytecodeRa),
        probe(JoltCommittedPolynomial::RamRa),
    )
    .unwrap();
    let dimensions = BooleanityDimensions::new(layout, log_t, log_k_chunk as usize);
    f(&backend, dimensions)
}

pub(crate) fn test_challenge(round: usize) -> Fr {
    use jolt_field::Ring;
    Fr::from_u64(0x1234_5678 + 1000 * round as u64 + 7)
}
