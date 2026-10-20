#![expect(
    clippy::expect_used,
    reason = "test scaffolding: fixture errors fail loudly"
)]

use std::ops::Range;

use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::{
    JoltAdviceKind, JoltCommittedPolynomial, JoltPolynomialId, PrecommittedClaimReduction,
};
use jolt_field::{Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::instructions::VirtualAdvice;
use jolt_riscv::{
    JoltInstruction, JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT,
};
use jolt_riscv::{RamRead, RamWrite};
use jolt_sumcheck::ProveRounds;
use jolt_witness::{
    ChunkVisitor, FixedBackend, JoltWitnessOracle, PolynomialEncoding, ProgramSource, RowSource,
    Shape, WitnessError,
};
use proptest::prelude::*;
use tracer::instruction::lw::LW;
use tracer::instruction::sw::SW;
use tracer::instruction::Cycle;
use tracer::instruction::RISCVCycle;

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
    use jolt_riscv::RamAccess;

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
            bytecode_words: words,
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
    }
}
