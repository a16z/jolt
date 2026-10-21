#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test scaffolding: device operations and fixture errors fail loudly"
)]

use std::ops::Range;
use std::sync::Arc;

use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::{
    JoltAdviceKind, JoltChallengeId, JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId,
    PrecommittedClaimReduction,
};
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::{Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::instructions::VirtualAdvice;
use jolt_riscv::{
    JoltInstruction, JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT,
};
use jolt_riscv::{RamAccess, RamRead, RamWrite, RegisterRead, RegisterState, RegisterWrite};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_witness::__private::TraceRow;
use jolt_witness::{
    ChunkVisitor, FixedBackend, JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle,
    JoltWitnessPlane, PolynomialEncoding, ProgramSource, RowSource, Shape, TraceBackend,
    WitnessError,
};
use proptest::prelude::*;
use tracer::instruction::lw::LW;
use tracer::instruction::sw::SW;
use tracer::instruction::Cycle;
use tracer::instruction::RISCVCycle;

use crate::reference::ReferenceBackend;
use crate::{PrepareKernel, ProofSession, ProverInputs};

pub fn fr(seed: u64) -> Fr {
    Fr::from_u64(seed.wrapping_mul(2_654_435_761) % 1_000_003 + 1)
}

pub fn arb_point(len: usize) -> impl Strategy<Value = Vec<Fr>> {
    proptest::collection::vec(any::<u64>().prop_map(fr), len)
}

pub fn ram_read_cycle(address: u64, value: u64) -> Cycle {
    Cycle::LW(RISCVCycle::<LW> {
        ram_access: RamRead { address, value },
        ..Default::default()
    })
}

pub fn ram_write_cycle(address: u64, pre_value: u64, post_value: u64) -> Cycle {
    Cycle::SW(RISCVCycle::<SW> {
        ram_access: RamWrite {
            address,
            pre_value,
            post_value,
        },
        ..Default::default()
    })
}

pub fn ram_trace(log_t: usize, ram_k: usize) -> Vec<Cycle> {
    (0..1usize << log_t)
        .map(|cycle| {
            let word = 1 + (cycle as u64 * 5) % (ram_k as u64 - 1);
            let address = 8 * word;
            match cycle % 4 {
                0 => Cycle::NoOp,
                1 => ram_read_cycle(address, 900 + cycle as u64),
                2 => ram_write_cycle(address, 100 + cycle as u64, 700 + cycle as u64),
                _ => ram_write_cycle(address, 400 + cycle as u64, 400 + cycle as u64),
            }
        })
        .collect()
}

pub struct RamCell {
    pub row: usize,
    pub col: usize,
    pub prev_val: u64,
    pub next_val: u64,
}

pub fn ram_matrix_cells(log_t: usize, ram_k: usize) -> Vec<RamCell> {
    let layout = MemoryLayout::default();
    let cells: Vec<RamCell> = ram_trace(log_t, ram_k)
        .iter()
        .enumerate()
        .filter_map(|(row, cycle)| {
            let (address, prev_val, next_val) = match cycle.ram_access() {
                RamAccess::Read(read) => (read.address, read.value, read.value),
                RamAccess::Write(write) => (write.address, write.pre_value, write.post_value),
                RamAccess::NoOp => return None,
            };
            let col = layout
                .remap_word_address(address)
                .expect("fixture RAM addresses are mapped")?;
            Some(RamCell {
                row,
                col: col as usize,
                prev_val,
                next_val,
            })
        })
        .collect();
    assert!(
        !cells.is_empty(),
        "the RAM fixture produced no accesses to build a matrix from",
    );
    cells
}

pub(crate) fn synthetic_row(
    mut instruction: JoltInstructionRow,
    mut registers: RegisterState,
    access: RamAccess,
    pc: u32,
) -> TraceRow {
    match access {
        RamAccess::Read(read) => {
            instruction.instruction_kind = JoltInstructionKind::LD;
            instruction.operands.rs2 = None;
            instruction.operands.rd = Some(1);
            registers.rs2 = None;
            registers.rd = Some(RegisterWrite {
                register: 1,
                pre_value: registers.rd.map_or(0, |write| write.pre_value),
                post_value: read.value,
            });
        }
        RamAccess::Write(write) => {
            instruction.instruction_kind = JoltInstructionKind::SD;
            instruction.operands.rd = None;
            instruction.operands.rs2 = Some(3);
            registers.rd = None;
            registers.rs2 = Some(RegisterRead {
                register: 3,
                value: write.post_value,
            });
        }
        RamAccess::NoOp => {
            if instruction.instruction_kind == JoltInstructionKind::NoOp {
                if registers == RegisterState::default() {
                    return TraceRow::default();
                }
                instruction.instruction_kind = JoltInstructionKind::ADD;
            }
        }
    }
    instruction.operands.rs1 = registers.rs1.map(|read| read.register);
    instruction.operands.rs2 = registers.rs2.map(|read| read.register);
    instruction.operands.rd = registers.rd.map(|write| write.register);
    TraceRow::new(instruction, registers, access, pc.max(1)).expect("checked synthetic row")
}

const RAM_FIXTURE_PATTERN: usize = 6;

const fn ram_fixture_padding(log_t: usize) -> usize {
    let cycles = 1usize << log_t;
    if cycles < 8 {
        1
    } else {
        cycles / 8
    }
}

pub const fn ram_fixture_is_cold(log_t: usize, cycle: usize) -> bool {
    cycle >= (1usize << log_t) - ram_fixture_padding(log_t)
        || matches!(cycle % RAM_FIXTURE_PATTERN, 0 | 4 | 5)
}

fn ram_rows(
    instruction: JoltInstructionRow,
    log_t: usize,
    layout: &MemoryLayout,
    ram_k: usize,
    seed: u64,
) -> Vec<TraceRow> {
    let cycles = 1usize << log_t;
    let padding_start = cycles - ram_fixture_padding(log_t);
    let lowest = layout.get_lowest_address();
    let below_lowest = lowest.saturating_sub(8);
    let mut last_hot = lowest + 8;
    let mut rows = Vec::with_capacity(cycles);

    for cycle in 0..cycles {
        let mix = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(cycle as u64 + 1);
        let word = (mix.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (mix >> 29)) % ram_k as u64;
        let address = lowest + 8 * word;
        let value = 900 + cycle as u64;

        let access = if cycle >= padding_start {
            RamAccess::NoOp
        } else if ram_fixture_is_cold(log_t, cycle) {
            match cycle % RAM_FIXTURE_PATTERN {
                4 => RamAccess::Read(RamRead { address: 0, value }),
                5 => RamAccess::Read(RamRead {
                    address: below_lowest,
                    value,
                }),
                _ => RamAccess::NoOp,
            }
        } else {
            match cycle % RAM_FIXTURE_PATTERN {
                2 => {
                    last_hot = address;
                    RamAccess::Write(RamWrite {
                        address,
                        pre_value: value,
                        post_value: value + 1,
                    })
                }
                3 => RamAccess::Read(RamRead {
                    address: last_hot,
                    value,
                }),
                _ => {
                    last_hot = address;
                    RamAccess::Read(RamRead { address, value })
                }
            }
        };

        rows.push(synthetic_row(
            instruction,
            RegisterState::default(),
            access,
            1,
        ));
    }
    rows
}

pub fn with_ram_witness<R>(
    log_t: usize,
    ram_k: usize,
    one_hot: JoltOneHotConfig,
    seed: u64,
    body: impl FnOnce(&TraceBackend) -> R,
) -> R {
    let instruction = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::XOR,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: Some(3),
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    };
    let memory_layout = MemoryLayout::new(&MemoryConfig {
        program_size: Some(1 << 12),
        ..MemoryConfig::default()
    });
    let preprocessing = Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(
            vec![instruction],
            instruction.address as u64,
            RV64IMAC_JOLT,
        )
        .expect("ram bytecode fixture"),
        ram: RAMPreprocessing::default(),
        memory_layout: memory_layout.clone(),
        max_padded_trace_length: 1usize << log_t,
    });
    let program = Arc::new(JoltProgram::default());
    let trace = TraceOutput::new(
        TraceData::new(ram_rows(instruction, log_t, &memory_layout, ram_k, seed)),
        Default::default(),
        None,
        None,
    );
    let backend = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, ram_k, one_hot),
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );
    body(&backend)
}

pub struct FixedPlane {
    columns: FixedBackend<Fr>,
    program: JoltProgramPreprocessing,
    label: &'static str,
    #[expect(dead_code, reason = "advice fixtures only expose oracle columns")]
    log_t: Option<usize>,
}

impl FixedPlane {
    pub fn with_log_t(
        columns: FixedBackend<Fr>,
        label: &'static str,
        log_t: Option<usize>,
    ) -> Self {
        Self::with_program(
            columns,
            label,
            log_t,
            JoltProgramPreprocessing {
                bytecode: BytecodePreprocessing::preprocess(
                    vec![JoltInstructionRow {
                        instruction_kind: JoltInstructionKind::ADD,
                        address: 0x8000_0000,
                        ..Default::default()
                    }],
                    0x8000_0000,
                    RV64IMAC_JOLT,
                )
                .expect("bytecode fixture"),
                ram: RAMPreprocessing::default(),
                memory_layout: MemoryLayout::default(),
                max_padded_trace_length: 1,
            },
        )
    }

    pub fn with_program(
        columns: FixedBackend<Fr>,
        label: &'static str,
        log_t: Option<usize>,
        program: JoltProgramPreprocessing,
    ) -> Self {
        Self {
            columns,
            program,
            label,
            log_t,
        }
    }
}

impl JoltWitnessOracle<Fr> for FixedPlane {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        self.columns.shape(id)
    }

    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<Fr>, WitnessError> {
        self.columns.oracle_table(id)
    }

    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        self.columns.committed_order()
    }
}

impl RowSource for FixedPlane {
    fn visit_chunks(
        &self,
        _range: Range<usize>,
        _chunk_size: usize,
        _visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        Err(WitnessError::InvalidWitnessData {
            label: self.label,
            reason: "this relation's fixture serves oracle columns only, not trace rows".to_owned(),
        })
    }
}

impl ProgramSource for FixedPlane {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        &self.program
    }
}

pub fn reference_input_claim<'a, R>(
    witness: &dyn JoltWitnessPlane<Fr>,
    make_inputs: impl Fn() -> ProverInputs<'a, Fr, R>,
) -> Fr
where
    R: ConcreteSumcheck<Fr> + 'a,
    ReferenceBackend: PrepareKernel<Fr, R>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr>,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let mut probe = ReferenceBackend
        .prepare(&mut ProofSession::default(), witness, make_inputs())
        .expect("reference prepare for the input-claim probe");
    probe_input_claim(&mut *probe)
}

pub fn probe_input_claim<K: ProveRounds<Fr> + ?Sized>(kernel: &mut K) -> Fr {
    kernel.prove_round(None, 0, Fr::from_u64(0)).map_or_else(
        |error| claim_from_round_check(&error),
        |poly| poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1)),
    )
}

fn claim_from_round_check(error: &SumcheckError<Fr>) -> Fr {
    match error {
        SumcheckError::RoundCheckFailed { actual, .. } => *actual,
        other => panic!("reference kernel failed on the fixture: {other:?}"),
    }
}

pub fn drive<K: ProveRounds<Fr> + ?Sized>(
    kernel: &mut K,
    input_claim: Fr,
    challenges: &[Fr],
) -> Vec<UnivariatePoly<Fr>> {
    let mut polys = Vec::new();
    let mut claim = input_claim;
    let mut bind = None;
    for (round, &challenge) in challenges.iter().enumerate() {
        let poly = kernel
            .prove_round(bind, round, claim)
            .expect("prove_round must succeed");
        claim = poly.evaluate(challenge);
        polys.push(poly);
        bind = Some(challenge);
    }
    kernel
        .finish_rounds(challenges[challenges.len() - 1])
        .expect("finish_rounds must succeed");
    polys
}

pub struct AdviceFixture {
    pub plane: FixedPlane,
    pub trusted: Vec<Fr>,
    pub untrusted: Vec<Fr>,
}

pub fn advice_plane(advice_bytes: usize, seed: u64) -> AdviceFixture {
    let words = (advice_bytes / 8).next_power_of_two().max(1);
    let log_words = words.trailing_zeros() as usize;
    let column = |salt: u64| -> Vec<Fr> {
        (0..words)
            .map(|index| fr(seed ^ salt.wrapping_mul(index as u64 + 1).wrapping_add(salt)))
            .collect()
    };
    let trusted = column(0x9E37_79B9_7F4A_7C15);
    let untrusted = column(0x85EB_CA6B_C2B2_AE35);

    let mut backend = FixedBackend::new();
    for (kind, values) in [
        (JoltAdviceKind::Trusted, &trusted),
        (JoltAdviceKind::Untrusted, &untrusted),
    ] {
        backend
            .insert(
                ram_val_check_advice_opening(kind).polynomial_id(),
                Shape::new(log_words, PolynomialEncoding::Dense),
                values.clone(),
            )
            .expect("insert advice column");
    }
    AdviceFixture {
        plane: FixedPlane::with_log_t(backend, "cuda advice_claim_reduction fixture", None),
        trusted,
        untrusted,
    }
}

pub fn precommitted_synthetic_point(len: usize, seed: u64) -> Vec<Fr> {
    (0..len)
        .map(|index| {
            fr(seed
                ^ 0x2545_F491_4F6C_DD1D_u64
                    .wrapping_mul(index as u64 + 1)
                    .wrapping_add(3))
        })
        .collect()
}

pub fn precommitted_round_challenges(rounds: usize, seed: u64) -> Vec<Fr> {
    (0..rounds)
        .map(|round| {
            fr(seed
                ^ 0x27D4_EB2F_1656_67C5_u64
                    .wrapping_mul(round as u64 + 1)
                    .wrapping_add(11))
        })
        .collect()
}

pub fn precommitted_cycle_variables(reduction: &PrecommittedClaimReduction, seed: u64) -> Vec<Fr> {
    let challenges = precommitted_round_challenges(reduction.cycle_phase_total_rounds(), seed);
    reduction
        .cycle_phase_variable_challenges(&challenges)
        .expect("cycle-phase variable challenges")
}

fn r1cs_fixture_operands(kind: JoltInstructionKind, slot: usize) -> NormalizedOperands {
    let magnitude = 13 + 5 * slot as i128;
    let imm = if slot.is_multiple_of(2) {
        -magnitude
    } else {
        magnitude
    };
    let (rd, rs1, rs2) = match kind {
        JoltInstructionKind::SD => (None, Some(2), Some(3)),
        JoltInstructionKind::BEQ => (None, Some(2), Some(3)),
        JoltInstructionKind::VirtualAssertEQ => (None, Some(2), Some(3)),
        JoltInstructionKind::LD => (Some(1), Some(2), None),
        JoltInstructionKind::ADDI
        | JoltInstructionKind::VirtualMULI
        | JoltInstructionKind::VirtualMovsign => (Some(1), Some(2), None),
        JoltInstructionKind::LUI | JoltInstructionKind::JAL => (Some(1), None, None),
        _ if kind == JoltInstruction::VirtualAdvice(VirtualAdvice(())) => (Some(1), None, None),
        _ => (Some(1), Some(2), Some(3)),
    };
    NormalizedOperands { rd, rs1, rs2, imm }
}

fn committed_bytecode(rows: usize) -> Vec<JoltInstructionRow> {
    let kinds = [
        JoltInstructionKind::XOR,
        JoltInstructionKind::ADD,
        JoltInstructionKind::SUB,
        JoltInstructionKind::MUL,
        JoltInstructionKind::LD,
        JoltInstructionKind::SD,
        JoltInstructionKind::JAL,
        JoltInstructionKind::BEQ,
        JoltInstructionKind::ADDI,
        JoltInstructionKind::LUI,
    ];
    (0..rows)
        .map(|slot| {
            let instruction_kind = kinds[slot % kinds.len()];
            JoltInstructionRow {
                instruction_kind,
                address: 0x8000_0000 + 4 * slot,
                operands: r1cs_fixture_operands(instruction_kind, slot),
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            }
        })
        .collect()
}

pub struct CommittedProgramFixture {
    pub plane: FixedPlane,
    pub bytecode_len: usize,
    pub image_words: Vec<u64>,
}

pub fn committed_program_plane(
    bytecode_rows: usize,
    image_words: usize,
    min_bytecode_address: u64,
    seed: u64,
) -> CommittedProgramFixture {
    let bytecode = BytecodePreprocessing::preprocess(
        committed_bytecode(bytecode_rows),
        0x8000_0000,
        RV64IMAC_JOLT,
    )
    .expect("committed bytecode fixture");
    let bytecode_len = bytecode.bytecode.len();
    let words: Vec<u64> = (0..image_words)
        .map(|index| {
            seed.wrapping_mul(0x2545_F491_4F6C_DD1D)
                ^ (index as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(index as u64 + 1)
        })
        .collect();
    let program = JoltProgramPreprocessing {
        bytecode,
        ram: RAMPreprocessing {
            min_bytecode_address,
            bytecode_words: words.clone(),
        },
        memory_layout: MemoryLayout::default(),
        max_padded_trace_length: 1,
    };
    CommittedProgramFixture {
        plane: FixedPlane::with_program(
            FixedBackend::new(),
            "cuda committed-program claim-reduction fixture",
            None,
            program,
        ),
        bytecode_len,
        image_words: words,
    }
}
