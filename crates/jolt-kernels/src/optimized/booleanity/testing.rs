use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId};
use jolt_field::Fr;
use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
use jolt_program::expand::{expand_instruction, ExpansionAllocator};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{
    CircuitFlags, Flags, JoltInstruction, JoltInstructionKind, JoltInstructionRow,
    NormalizedOperands, SourceInstruction, SourceInstructionKind, SourceInstructionRow,
    RV64IMAC_JOLT,
};
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

/// The halt that ends a run.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Halt {
    /// `jolt.success`.
    Success,
    /// `EBREAK`.
    Panic,
}

impl Halt {
    /// The bytecode row that expansion lowers the decoded instruction to: one
    /// row that writes its link to a temporary.
    fn instruction(self) -> JoltInstructionRow {
        let (kind, address, imm) = match self {
            Self::Success => (SourceInstructionKind::JoltSuccess, 0x8000_000c, 0),
            Self::Panic => (SourceInstructionKind::EBREAK, 0x8000_0010, 1),
        };
        let source = SourceInstruction::new(
            kind,
            SourceInstructionRow {
                address,
                operands: NormalizedOperands {
                    rd: Some(0),
                    rs1: Some(0),
                    rs2: None,
                    imm,
                },
                ..SourceInstructionRow::default()
            },
        );
        let [row] = expand_instruction(&source, &mut ExpansionAllocator::new(), RV64IMAC_JOLT)
            .unwrap()
            .try_into()
            .unwrap();
        row.into()
    }
}

/// Runs `f` against a trace backend over a run of `len` rows that ends at
/// `halt`: loads, ALU rows and stores, then the halt, after which only the
/// backend's padding up to `2^log_t` follows, as in a proved run. The program
/// holds both halts, so the bytecode read-RAF stage values fold `Halt` and
/// `HaltPanic` whichever halt ends the run.
pub(crate) fn with_halting_backend<R>(
    log_t: usize,
    log_k_chunk: u8,
    len: usize,
    halt: Halt,
    f: impl FnOnce(&TraceBackend, BooleanityDimensions) -> R,
) -> R {
    let halts = [Halt::Success, Halt::Panic].map(Halt::instruction);
    for (instruction, panics) in halts.iter().zip([false, true]) {
        let flags = JoltInstruction::from(*instruction).circuit_flags();
        assert!(flags[CircuitFlags::Halt] && flags[CircuitFlags::HaltPanic] == panics);
    }
    let bytecode = fixture_bytecode([LOAD, STORE, ALU].into_iter().chain(halts).collect());
    let instruction = halt.instruction();
    let link = instruction.address as u64 + 4;
    let mut rows: Vec<TraceRow> = (0..len as u64 - 1)
        .map(|cycle| {
            let address = 0x8000_1000 + 8 * (cycle % 11);
            match cycle % 3 {
                0 => load_row(address),
                1 => alu_row(cycle.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                _ => store_row(address),
            }
        })
        .collect();
    rows.push(
        TraceRow::new(
            instruction,
            RegisterState {
                rs1: instruction
                    .operands
                    .rs1
                    .map(|register| RegisterRead { register, value: 0 }),
                rd: instruction.operands.rd.map(|register| RegisterWrite {
                    register,
                    pre_value: 0,
                    post_value: link,
                }),
                ..Default::default()
            },
            RamAccess::NoOp,
            bytecode.get_pc(&instruction).unwrap() as u32,
        )
        .unwrap(),
    );
    with_bytecode_backend(log_t, log_k_chunk, bytecode, rows, f)
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
    with_bytecode_backend(
        log_t,
        log_k_chunk,
        fixture_bytecode(vec![LOAD, STORE, ALU]),
        rows,
        f,
    )
}

fn fixture_bytecode(program: Vec<JoltInstructionRow>) -> BytecodePreprocessing {
    BytecodePreprocessing::preprocess(program, LOAD.address as u64, RV64IMAC_JOLT).unwrap()
}

fn with_bytecode_backend<R>(
    log_t: usize,
    log_k_chunk: u8,
    bytecode: BytecodePreprocessing,
    rows: Vec<TraceRow>,
    f: impl FnOnce(&TraceBackend, BooleanityDimensions) -> R,
) -> R {
    use std::sync::Arc;
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode,
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
