//! Shared field-inline trace fixtures for the stage-recipe round-trip tests.
//!
//! Hand-crafted rows that are semantically consistent instruction executions
//! (the same discipline as `jolt_witness::testing::with_sample_backend`), so
//! the composed R1CS eq rows are satisfied and the stage sumchecks' hard
//! self-checks hold — including the stage-4 register-file and RAM value
//! checks (consistent register reads, and the termination store the witness
//! plane's device-derived final RAM state demands). Two profiles: an
//! ADDI-only trace (a field-inline guest executing zero field-inline instructions —
//! every field-inline column is zero), and a field arithmetic trace (two field loads and
//! a multiply, the stage-0 fixture's rows) whose decoded field-inline instruction
//! words populate the field-inline columns.

#![expect(
    clippy::unwrap_used,
    reason = "hand-crafted fixture rows fail loudly when malformed"
)]

use std::sync::Arc;

use common::constants::{MAX_BLINDFOLD_GENERATORS, RAM_START_ADDRESS};
use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltRelationId};
#[cfg(feature = "zk")]
use jolt_crypto::{Bn254, JoltGroup, PedersenSetup};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
#[cfg(feature = "zk")]
use jolt_field::Ring;
use jolt_program::execution::{
    JoltProgram, OwnedTrace, RamAccess, RamWrite, RegisterRead, RegisterState, RegisterWrite,
    TraceOutput, TraceRow,
};
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{
    FieldInlineOp, JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow,
    NormalizedOperands, RV64IMAC_JOLT_FIELD_INLINE,
};
use jolt_transcript::{Channel, ProverTranscript, VerifierTranscript};
use jolt_verifier::preprocessing::{JoltVerifierPreprocessing, ProgramPreprocessing};
use jolt_verifier::stages::{
    build_formula_dimensions, stage1, stage2, stage3, stage4, stage5, stage6a, stage6b,
    PrecommittedSchedule,
};
use jolt_verifier::{jolt_protocol_id, CheckedInputs, JoltSponge, JOLT_SESSION};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::{JoltProverPreprocessing, ProverConfig};

pub(crate) const ENTRY: u64 = RAM_START_ADDRESS;
// 3, not 2: the last physical cycle must be a noop (constraint 21's
// ShouldJump convention), so the field-inline fixture's six real rows need padding
// room behind them.
pub(crate) const LOG_T: usize = 3;
// Matches the witness backend's `JoltVmWitnessConfig` ram size (64).
pub(crate) const RAM_LOG_K: usize = 6;

pub(crate) type FixturePreprocessing = JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

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

/// The fixture programs' preprocessing, shared verbatim between the witness
/// backend and the prover-preprocessing carrier so both fronts see the same
/// bytecode facts (PC mapping and canonical instruction operands).
#[expect(clippy::unwrap_used, reason = "test fixture construction")]
fn fixture_program_preprocessing(
    bytecode: Vec<JoltInstructionRow>,
) -> Arc<JoltProgramPreprocessing> {
    Arc::new(JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(bytecode, ENTRY, RV64IMAC_JOLT_FIELD_INLINE)
            .unwrap(),
        ram: RAMPreprocessing::default(),
        memory_layout: test_memory_layout(),
        max_padded_trace_length: 1 << LOG_T,
    })
}

pub(crate) fn field_inline_backend(
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
    let preprocessing = fixture_program_preprocessing(bytecode);
    TraceBackend::new(
        JoltVmWitnessConfig::new(
            LOG_T,
            1 << RAM_LOG_K,
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
    let mut row = TraceRow::from_instruction(instruction).unwrap();
    row.field_inline = Some(data.into());
    row
}

/// A terminal JAL row: the only hand-craftable last real instruction — its
/// `Jump` flag turns off the otherwise-unconditional PC-update row 16, and
/// `ShouldJump` stays 0 because the successor is the noop padding — with the
/// link write (`rd = address + 4`) row 13 demands.
fn halt_jal_row(offset: usize, rd: u8) -> TraceRow {
    let jal = instruction(JoltInstructionKind::JAL, offset, Some(rd), None, None, 0);
    TraceRow::new(
        jal,
        RegisterState {
            rd: Some(RegisterWrite {
                register: rd,
                pre_value: 0,
                post_value: ENTRY + (offset as u64) * 4 + 4,
            }),
            ..Default::default()
        },
        RamAccess::NoOp,
    )
    .unwrap()
}

/// The guest termination convention, hand-crafted: the witness plane's final
/// RAM state unconditionally carries `termination = 1` (a real guest writes
/// it before halting), so any trace that must satisfy the stage-4 RAM value
/// check needs a matching increment. Two rows: `ADDI x6, x0, 1` (a consistent
/// register write of the stored value), then `SD x6, termination(x0)` (store
/// flag on, `RamAddress = rs1 + imm = termination`, `RamWriteValue = rs2`).
fn termination_store_rows(offset: usize) -> [TraceRow; 2] {
    let one = instruction(JoltInstructionKind::ADDI, offset, Some(6), Some(0), None, 1);
    let termination = test_memory_layout().termination;
    let store = instruction(
        JoltInstructionKind::SD,
        offset + 1,
        None,
        Some(0),
        Some(6),
        termination as i128,
    );
    [
        TraceRow::new(
            one,
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 0,
                    value: 0,
                }),
                rd: Some(RegisterWrite {
                    register: 6,
                    pre_value: 0,
                    post_value: 1,
                }),
                ..Default::default()
            },
            RamAccess::NoOp,
        )
        .unwrap(),
        TraceRow::new(
            store,
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 0,
                    value: 0,
                }),
                rs2: Some(RegisterRead {
                    register: 6,
                    value: 1,
                }),
                ..Default::default()
            },
            RamAccess::Write(RamWrite {
                address: termination,
                pre_value: 0,
                post_value: 1,
            }),
        )
        .unwrap(),
    ]
}

/// A field-inline guest executing only ordinary instructions (an ADDI with
/// consistent register semantics, the termination store, then the terminal
/// JAL): the rv64 eq rows are satisfied while every field-inline column is zero.
fn addi_only_program() -> (Vec<JoltInstructionRow>, Vec<TraceRow>) {
    let addi = instruction(JoltInstructionKind::ADDI, 0, Some(1), Some(2), None, 3);
    let [one, store] = termination_store_rows(1);
    let jal = halt_jal_row(3, 5);
    let rows = vec![
        TraceRow::new(
            addi,
            RegisterState {
                // Register 2 is never written, so the read must see the
                // initial value — the stage-4 register file check binds it.
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 0,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 0,
                    post_value: 3,
                }),
                ..Default::default()
            },
            RamAccess::NoOp,
        )
        .unwrap(),
        one.clone(),
        store.clone(),
        jal.clone(),
    ];
    (
        vec![
            addi,
            one.instruction(),
            store.instruction(),
            jal.instruction(),
        ],
        rows,
    )
}

pub(crate) fn addi_only_backend() -> TraceBackend<OwnedTrace> {
    let (bytecode, rows) = addi_only_program();
    field_inline_backend(bytecode, rows)
}

/// Two field loads and a multiply: `FieldRdInc = [13, 17, 221, 0]`,
/// `13 · 17 = 221` — every field-inline eq row and both field-inline product lanes are satisfied
/// (the product columns are extractor-derived), and the x-register file is
/// untouched.
fn field_arithmetic_program() -> (Vec<JoltInstructionRow>, Vec<TraceRow>) {
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
    let [one, store] = termination_store_rows(3);
    let jal = halt_jal_row(5, 5);
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
                ..FieldInlineTraceData::default()
            },
        ),
        one.clone(),
        store.clone(),
        jal.clone(),
    ];
    (
        vec![
            load_a,
            load_b,
            mul,
            one.instruction(),
            store.instruction(),
            jal.instruction(),
        ],
        rows,
    )
}

pub(crate) fn field_arithmetic_backend() -> TraceBackend<OwnedTrace> {
    let (bytecode, rows) = field_arithmetic_program();
    field_inline_backend(bytecode, rows)
}

/// The prover-preprocessing carrier the stage-4+ recipes take, over the
/// fixture program: a full-program verifier preprocessing (the same
/// `JoltProgramPreprocessing` the witness backend holds) and a minimal Dory
/// setup — the reference-tier stage recipes never commit through it.
fn prover_preprocessing(bytecode: Vec<JoltInstructionRow>) -> FixturePreprocessing {
    JoltProverPreprocessing {
        verifier: JoltVerifierPreprocessing::new(
            ProgramPreprocessing::Full(fixture_program_preprocessing(bytecode)),
            DoryScheme::setup_verifier(2),
            None,
        )
        .unwrap(),
        pcs_setup: DoryScheme::setup_prover(2),
        committed_program: None,
    }
}

pub(crate) fn field_arithmetic_preprocessing() -> FixturePreprocessing {
    prover_preprocessing(field_arithmetic_program().0)
}

pub(crate) fn addi_only_preprocessing() -> FixturePreprocessing {
    prover_preprocessing(addi_only_program().0)
}

/// The stage-4+ recipes' checked-inputs carrier for the fixture traces,
/// mirroring what shape validation derives for a field-inline proof at this scale
/// (no advice, no precommitted objects, full program) under
/// [`test_prover_config`]'s shape.
pub(crate) fn test_checked_inputs() -> CheckedInputs {
    let config = test_prover_config();
    CheckedInputs {
        public_io: test_public_io(),
        zk: cfg!(feature = "zk"),
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
        untrusted_advice_commitment_present: false,
        entry_address: ENTRY,
        preprocessing_digest: [0u8; 32],
        trusted_advice_commitment_present: false,
        vc_capacity: cfg!(feature = "zk").then_some(MAX_BLINDFOLD_GENERATORS),
        precommitted: PrecommittedSchedule {
            #[cfg(not(feature = "akita"))]
            trusted_advice: None,
            #[cfg(not(feature = "akita"))]
            untrusted_advice: None,
            bytecode: None,
            program_image: None,
        },
    }
}

/// The stage recipes' derived-config shape for the fixture traces: the same
/// derivation `ProverConfig::derive` performs, at the fixture's scale (no
/// RAM traffic, so `ram_K` stays at a small power of two).
pub(crate) fn test_prover_config() -> ProverConfig {
    ProverConfig {
        trace_length: 1 << LOG_T,
        ram_K: 1 << RAM_LOG_K,
        rw_config: crate::config::read_write_config(LOG_T, RAM_LOG_K),
        one_hot_config: crate::config::one_hot_config(LOG_T),
        trace_polynomial_order: Default::default(),
    }
}

/// A well-formed memory layout for the fixture traces (the default layout is
/// degenerate: its lowest mapped address is zero, which `PublicIoMemory`
/// rejects).
pub(crate) fn test_memory_layout() -> MemoryLayout {
    MemoryLayout::new(&MemoryConfig {
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
pub(crate) fn test_public_io() -> JoltDevice {
    JoltDevice {
        memory_layout: test_memory_layout(),
        ..Default::default()
    }
}

/// The BlindFold row-commitment setup for the committed stage recipes:
/// `MAX_BLINDFOLD_GENERATORS` distinct generator multiples, matching
/// [`test_checked_inputs`]'s `vc_capacity`.
#[cfg(feature = "zk")]
pub(crate) fn test_vc_setup() -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let generators = (2..2 + MAX_BLINDFOLD_GENERATORS as u64)
        .map(|k| generator.scalar_mul(&Fr::from_u64(k)))
        .collect();
    PedersenSetup::new(generators, generator.scalar_mul(&Fr::from_u64(1)))
}

/// A fresh prover transcript under Jolt's protocol id and session. The
/// fixture prover starts at stage 1 (no stage 0), so [`verify_through`]
/// runs the verifier stages from stage 1 on the same fresh transcript.
pub(crate) fn fixture_transcript() -> ProverTranscript<JoltSponge> {
    ProverTranscript::new(&jolt_protocol_id::<JoltSponge>(), JOLT_SESSION)
}

/// The last production verifier stage [`verify_through`] runs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Through {
    Stage1,
    Stage2,
    Stage4,
    Stage5,
    Stage6a,
    Stage6b,
}

/// Verifies the fixture prover's argument string with the production stage
/// verifiers through `last`, requiring the verifier to consume every byte
/// and land on the prover's sponge state.
pub(crate) fn verify_through(
    last: Through,
    checked: &CheckedInputs,
    preprocessing: &FixturePreprocessing,
    prover: &mut ProverTranscript<JoltSponge>,
) {
    let narg = prover.narg().to_vec();
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &narg,
    );
    let verifier = &preprocessing.verifier;
    let formula_dimensions =
        build_formula_dimensions(verifier, checked, LOG_T, JoltRelationId::InstructionReadRaf)
            .unwrap();
    'stages: {
        let t = &mut transcript;
        let stage1 = stage1::verify::<Fr, Bn254G1, JoltSponge>(checked, t).unwrap();
        if last == Through::Stage1 {
            break 'stages;
        }
        let stage2 = stage2::verify(checked, t, &stage1).unwrap();
        if last == Through::Stage2 {
            break 'stages;
        }
        let stage3 = stage3::verify(checked, t, &stage1, &stage2).unwrap();
        let stage4 = stage4::verify(checked, verifier, t, &stage2, &stage3).unwrap();
        if last == Through::Stage4 {
            break 'stages;
        }
        let stage5 = stage5::verify(checked, &formula_dimensions, t, &stage2, &stage4).unwrap();
        if last == Through::Stage5 {
            break 'stages;
        }
        let stage6a = stage6a::verify(
            checked,
            verifier,
            &formula_dimensions,
            t,
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &stage5,
        )
        .unwrap();
        if last == Through::Stage6a {
            break 'stages;
        }
        let _stage6b = stage6b::verify(
            checked,
            verifier,
            &formula_dimensions,
            t,
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &stage5,
            &stage6a,
        )
        .unwrap();
    }
    assert_eq!(transcript.remaining(), 0, "unconsumed argument bytes");
    assert_eq!(
        transcript.challenge_bytes::<32>(),
        prover.challenge_bytes::<32>(),
        "verifier and prover sponge states diverge"
    );
    transcript.finish().unwrap();
}

/// Shared upstream proving for clear and committed stage tests.
pub(crate) mod proving {
    use super::*;
    use crate::stages::stage1::{prove_stage1, Stage1ProverOutput};
    use crate::stages::stage2::{prove_stage2, Stage2ProverOutput};
    use crate::stages::stage3::{prove_stage3, Stage3ProverOutput};
    use crate::stages::stage4::{prove_stage4, Stage4ProverOutput};
    use crate::stages::stage5::{prove_stage5, Stage5ProverOutput};
    use crate::stages::stage6a::{prove_stage6a, Stage6aProverOutput};
    use crate::{JoltBackend, ProofMode};
    use jolt_kernels::ProofSession;
    use jolt_witness::JoltWitnessPlane;
    type Stages3 = (
        Stage1ProverOutput<Fr>,
        Stage2ProverOutput<Fr>,
        Stage3ProverOutput<Fr>,
    );
    type Stages4 = (Stages3, Stage4ProverOutput<Fr>);
    type Stages5 = (Stages4, Stage5ProverOutput<Fr>);
    type Stages6a = (Stages5, Stage6aProverOutput<Fr>);

    pub(crate) struct FixtureProver<'a> {
        pub(crate) backend: &'a JoltBackend<Fr, DoryScheme>,
        pub(crate) session: &'a mut ProofSession,
        pub(crate) mode: &'a ProofMode<'a, Pedersen<Bn254G1>>,
        pub(crate) config: &'a ProverConfig,
        pub(crate) public_io: &'a JoltDevice,
        pub(crate) checked: &'a CheckedInputs,
        pub(crate) preprocessing: &'a FixturePreprocessing,
        pub(crate) witness: &'a dyn JoltWitnessPlane<Fr>,
        pub(crate) transcript: &'a mut ProverTranscript<JoltSponge>,
    }

    impl FixtureProver<'_> {
        pub(crate) fn through_stage3(&mut self) -> Stages3 {
            let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                self.backend,
                self.session,
                self.mode,
                LOG_T,
                self.witness,
                self.transcript,
            )
            .unwrap();
            let stage2 = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                self.backend,
                self.session,
                self.mode,
                self.config,
                self.public_io,
                &stage1.clear_output,
                self.witness,
                self.transcript,
            )
            .unwrap();
            let stage3 = prove_stage3::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                self.backend,
                self.session,
                self.mode,
                self.config,
                &stage1.clear_output,
                &stage2.clear_output,
                self.witness,
                self.transcript,
            )
            .unwrap();
            (stage1, stage2, stage3)
        }
        pub(crate) fn through_stage4(&mut self) -> Stages4 {
            let (stage1, stage2, stage3) = self.through_stage3();
            let stage4 = prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                self.backend,
                self.session,
                self.mode,
                self.checked,
                self.config,
                self.preprocessing,
                &stage2.clear_output,
                &stage3.clear_output,
                self.witness,
                self.transcript,
            )
            .unwrap();
            ((stage1, stage2, stage3), stage4)
        }
        pub(crate) fn through_stage5(&mut self) -> Stages5 {
            let ((stage1, stage2, stage3), stage4) = self.through_stage4();
            let stage5 = prove_stage5::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                self.backend,
                self.session,
                self.mode,
                self.checked,
                self.config,
                self.preprocessing,
                &stage2.clear_output,
                &stage4.clear_output,
                self.witness,
                self.transcript,
            )
            .unwrap();
            (((stage1, stage2, stage3), stage4), stage5)
        }
        pub(crate) fn through_stage6a(&mut self) -> Stages6a {
            let (((stage1, stage2, stage3), stage4), stage5) = self.through_stage5();
            let stage6a = prove_stage6a::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                self.backend,
                self.session,
                self.mode,
                self.checked,
                self.config,
                self.preprocessing,
                &stage1.clear_output,
                &stage2.clear_output,
                &stage3.clear_output,
                &stage4.clear_output,
                &stage5.clear_output,
                self.witness,
                self.transcript,
            )
            .unwrap();
            ((((stage1, stage2, stage3), stage4), stage5), stage6a)
        }
    }
}
