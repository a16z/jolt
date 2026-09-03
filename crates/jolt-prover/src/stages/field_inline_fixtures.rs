//! Shared FR-profile trace fixtures for the stage-recipe round-trip tests.
//!
//! Hand-crafted rows that are semantically consistent instruction executions
//! (the same discipline as `jolt_witness::testing::with_sample_backend`), so
//! the composed R1CS eq rows are satisfied and the stage sumchecks' hard
//! self-checks hold — including the stage-4 register-file and RAM value
//! checks (consistent register reads, and the termination store the witness
//! plane's device-derived final RAM state demands). Two profiles: an
//! ADDI-only trace (an FR-profile guest executing zero FR instructions —
//! every FR column is zero), and an FR arithmetic trace (two field loads and
//! a multiply, the stage-0 fixture's rows) whose decoded FR instruction
//! words populate the FR columns.

#![expect(
    clippy::unwrap_used,
    reason = "hand-crafted fixture rows fail loudly when malformed"
)]

use std::sync::Arc;

use common::constants::{MAX_BLINDFOLD_GENERATORS, RAM_START_ADDRESS, REGISTER_COUNT};
use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
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
use jolt_verifier::preprocessing::{JoltVerifierPreprocessing, ProgramPreprocessing};
#[cfg(feature = "akita")]
use jolt_verifier::stages::stage8::field_inline_packed::FieldIncLimbsScheduled;
use jolt_verifier::stages::PrecommittedSchedule;
use jolt_verifier::CheckedInputs;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::{JoltProverPreprocessing, ProverConfig};

pub(crate) const ENTRY: u64 = RAM_START_ADDRESS;
// 3, not 2: the last physical cycle must be a noop (constraint 21's
// ShouldJump convention), so the FR fixture's six real rows need padding
// room behind them.
pub(crate) const LOG_T: usize = 3;
// Matches the witness backend's `JoltVmWitnessConfig` ram size (64).
pub(crate) const RAM_LOG_K: usize = 6;

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
/// bytecode facts (PC mapping, FR side-table metadata).
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
    let preprocessing = fixture_program_preprocessing(bytecode);
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

/// An FR-profile guest executing only ordinary instructions (an ADDI with
/// consistent register semantics, the termination store, then the terminal
/// JAL): the rv64 eq rows are satisfied while every FR column is zero.
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
    fr_backend(bytecode, rows)
}

/// Two field loads and a multiply: `FieldRdInc = [13, 17, 221, 0]`,
/// `13 · 17 = 221` — every FR eq row and both FR product lanes are satisfied
/// (the product columns are extractor-derived), and the x-register file is
/// untouched.
fn fr_arithmetic_program() -> (Vec<JoltInstructionRow>, Vec<TraceRow>) {
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
                product: Some(enc(221)),
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

pub(crate) fn fr_arithmetic_backend() -> TraceBackend<OwnedTrace> {
    let (bytecode, rows) = fr_arithmetic_program();
    fr_backend(bytecode, rows)
}

/// The prover-preprocessing carrier the stage-4+ recipes take, over the
/// fixture program: a full-program verifier preprocessing (the same
/// `JoltProgramPreprocessing` the witness backend holds) and a minimal Dory
/// setup — the reference-tier stage recipes never commit through it.
fn prover_preprocessing(
    bytecode: Vec<JoltInstructionRow>,
) -> JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>> {
    JoltProverPreprocessing {
        verifier: JoltVerifierPreprocessing::new(
            ProgramPreprocessing::Full(fixture_program_preprocessing(bytecode)),
            [0u8; 32],
            DoryScheme::setup_verifier(2),
            None,
        ),
        pcs_setup: DoryScheme::setup_prover(2),
        committed_program: None,
    }
}

pub(crate) fn fr_arithmetic_preprocessing() -> JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>>
{
    prover_preprocessing(fr_arithmetic_program().0)
}

pub(crate) fn addi_only_preprocessing() -> JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>> {
    prover_preprocessing(addi_only_program().0)
}

/// The stage-4+ recipes' checked-inputs carrier for the fixture traces,
/// mirroring what shape validation derives for an FR-on proof at this scale
/// (no advice, no precommitted objects, full program).
pub(crate) fn test_checked_inputs() -> CheckedInputs {
    CheckedInputs {
        public_io: test_public_io(),
        zk: cfg!(feature = "zk"),
        trace_length: 1 << LOG_T,
        ram_K: 1 << RAM_LOG_K,
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
            #[cfg(feature = "akita")]
            field_inc_limbs: Some(FieldIncLimbsScheduled),
        },
    }
}

/// The stage recipes' derived-config shape for the fixture traces: the same
/// derivation `ProverConfig::derive` performs, at the fixture's scale (no
/// RAM traffic, so `ram_K` stays at a small power of two).
pub(crate) fn test_prover_config() -> ProverConfig {
    // Matches the witness backend's `JoltVmWitnessConfig` ram size (64).
    const RAM_LOG_K: usize = 6;
    ProverConfig {
        trace_length: 1 << LOG_T,
        ram_K: 1 << RAM_LOG_K,
        rw_config: JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: LOG_T as u8,
            ram_rw_phase2_num_rounds: RAM_LOG_K as u8,
            registers_rw_phase1_num_rounds: LOG_T as u8,
            registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
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

/// Twin-transcript replays of the already-round-tripped upstream stages, for
/// the downstream stage twins: each helper advances `transcript` exactly as
/// `stageN::verify`'s clear body does over the prover's outputs (with
/// `verify_clear` hard-checking the wire rounds on the way). The full
/// `verify` entrypoints need an assembled `JoltProof`, so the twins drive the
/// same public constituents instead — the stage-1/2 bodies are the ones
/// stage 2's own round-trip test pins.
#[cfg(not(feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test twin helpers")]
pub(crate) mod twins {
    use common::jolt_device::JoltDevice;
    use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
    use jolt_claims::protocols::jolt::geometry::spartan::{
        SpartanOuterDimensions, SpartanProductDimensions,
    };
    use jolt_claims::protocols::jolt::TraceDimensions;
    use jolt_claims::NoChallenges;
    use jolt_field::{Fr, Ring};
    use jolt_program::preprocess::PublicIoMemory;
    use jolt_transcript::{AppendToTranscript, LegacyBlake2bTranscript as Blake2bTranscript};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainder,
    };
    use jolt_verifier::stages::stage1::outputs::{Stage1BatchInputClaims, Stage1BatchSumchecks};
    use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
    use jolt_verifier::stages::stage2::outputs::Stage2BatchSumchecks;
    use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
    use jolt_verifier::stages::stage2::product_uniskip::{
        product_uniskip_input_values_from_stage1, ProductUniskip,
    };
    use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
    use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
    use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
    use jolt_verifier::stages::stage2::{
        field_inline as stage2_field_inline, product_tau_low,
        stage2_batch_input_values_from_upstream,
    };
    use jolt_verifier::stages::stage3::outputs::{
        InstructionInput, RegistersClaimReduction, SpartanShift, Stage3Sumchecks,
    };
    use jolt_verifier::stages::stage3::stage3_input_values_from_upstream;
    use jolt_verifier::stages::uniskip::{
        self, draw_spartan_outer_tau, draw_spartan_product_tau_high, UniskipParams,
    };

    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    #[cfg(feature = "akita")]
    use jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims;
    use jolt_claims::protocols::jolt::JoltRelationId;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_verifier::stages::stage4::outputs::Stage4Sumchecks;
    use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
    use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
    use jolt_verifier::stages::stage4::{
        field_inline as stage4_field_inline, public_initial_ram_evaluation,
        ram_val_check_init_structure, stage4_input_points_from_upstream,
        stage4_input_values_from_upstream, RamValCheckInitialEvaluation,
    };
    use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
    use jolt_verifier::stages::stage5::outputs::Stage5Sumchecks;
    use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
    use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
    use jolt_verifier::stages::stage5::{
        field_inline as stage5_field_inline, stage5_input_points_from_upstream,
        stage5_input_values_from_upstream,
    };
    use jolt_verifier::stages::stage6a::batch::Stage6aBuildParts;
    use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhaseInputClaims;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::bytecode_read_raf_address_phase_input_values_from_upstream;
    use jolt_verifier::stages::stage6a::field_inline as stage6a_field_inline;
    use jolt_verifier::stages::stage6a::outputs::{Stage6aInputClaims, Stage6aSumchecks};
    use jolt_verifier::CheckedInputs;

    use super::{LOG_T, RAM_LOG_K};
    use crate::stages::stage1::Stage1ProverOutput;
    use crate::stages::stage2::Stage2ProverOutput;
    use crate::stages::stage3::Stage3ProverOutput;
    use crate::stages::stage4::Stage4ProverOutput;
    use crate::stages::stage5::Stage5ProverOutput;
    use crate::stages::stage6a::Stage6aProverOutput;
    use crate::{JoltProverPreprocessing, ProverConfig};

    pub(crate) type FixturePreprocessing = JoltProverPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

    /// Stage 1's twin (already round-tripped by stage 1's own tests):
    /// positions the transcript at the stage-2 boundary.
    pub(crate) fn replay_stage1<C: Clone + AppendToTranscript>(
        transcript: &mut Blake2bTranscript,
        stage1: &Stage1ProverOutput<Fr, C>,
    ) {
        let tau = draw_spartan_outer_tau(transcript, LOG_T);
        let uniskip_challenge = uniskip::verify_clear(
            &stage1.uniskip_proof,
            &UniskipParams::spartan_outer(),
            Fr::from_u64(0),
            stage1.claims.uniskip_output_claim,
            transcript,
        )
        .unwrap();
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(
                SpartanOuterDimensions::rv64(LOG_T),
                tau,
                uniskip_challenge,
            ),
        };
        let batch_challenges = sumchecks.draw_challenges(transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        let attached = jolt_verifier::stages::stage1::field_inline::attach_outer_outputs(
            &sumchecks,
            &stage1.claims,
        )
        .unwrap();
        let input_values = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(
                stage1.claims.uniskip_output_claim,
            ),
        };
        let _stage1_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &batch_challenges,
                &stage1.claims.outer,
                &stage1.sumcheck_proof,
                transcript,
                1,
            )
            .unwrap();
        sumchecks.append_output_claims(transcript, &stage1.claims.outer);
        jolt_verifier::stages::stage1::field_inline::append_outer_openings(transcript, &attached);
    }

    /// Stage 2's twin (already round-tripped by stage 2's own tests):
    /// positions the transcript at the stage-3 boundary.
    pub(crate) fn replay_stage2<C: Clone + AppendToTranscript>(
        transcript: &mut Blake2bTranscript,
        config: &ProverConfig,
        public_io: &JoltDevice,
        stage1: &Stage1ProverOutput<Fr, C>,
        stage2: &Stage2ProverOutput<Fr, C>,
    ) {
        let log_t = LOG_T;
        let log_k = config.ram_K.ilog2() as usize;
        let trace_dimensions = TraceDimensions::new(log_t);
        let read_write_dimensions = config.rw_config.ram_dimensions(log_t, log_k);
        let product_dimensions = SpartanProductDimensions::new(log_t);
        let raf_dimensions = RamRafEvaluationDimensions::try_from(read_write_dimensions).unwrap();
        let tau_low = product_tau_low(&stage1.clear_output.remainder_point(), log_t).unwrap();

        let tau_high: Fr = draw_spartan_product_tau_high(transcript);
        let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
        stage2_field_inline::attach_uniskip_inputs(&uniskip_relation, &stage1.clear_output)
            .unwrap();
        let uniskip_inputs = product_uniskip_input_values_from_stage1(&stage1.clear_output);
        let uniskip_input_claim = uniskip_relation
            .input_claim(&uniskip_inputs, &NoChallenges::default())
            .unwrap();
        let uniskip_challenge = uniskip::verify_clear(
            &stage2.uniskip_proof,
            &UniskipParams::spartan_product(),
            uniskip_input_claim,
            stage2.claims.product_uniskip_output_claim,
            transcript,
        )
        .unwrap();

        let lowest_address = public_io.memory_layout.get_lowest_address();
        let public_memory = PublicIoMemory::new(public_io).unwrap();
        let sumchecks = Stage2BatchSumchecks {
            ram_read_write: RamReadWriteChecking::new(
                read_write_dimensions,
                log_k,
                tau_low.clone(),
            ),
            product_remainder: ProductRemainder::new(
                product_dimensions,
                uniskip_challenge,
                tau_high,
                tau_low.clone(),
            ),
            instruction_claim_reduction: InstructionClaimReduction::new(
                trace_dimensions,
                tau_low.clone(),
            ),
            field_registers_claim_reduction: stage2_field_inline::claim_reduction_member(
                log_t,
                tau_low.clone(),
            ),
            ram_raf_evaluation: RamRafEvaluation::new(
                read_write_dimensions,
                raf_dimensions,
                log_k,
                lowest_address,
                tau_low.clone(),
            ),
            ram_output_check: RamOutputCheck::new(read_write_dimensions, public_memory),
        };
        let challenges = sumchecks.draw_challenges(transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        sumchecks
            .validate_output_claims(&stage2.claims.batch_outputs)
            .unwrap();
        let attached_product =
            stage2_field_inline::attach_product_outputs(&sumchecks, &stage2.claims).unwrap();
        let input_values = stage2_batch_input_values_from_upstream(
            &stage1.clear_output,
            stage2.claims.product_uniskip_output_claim,
        )
        .unwrap();
        let _stage2_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &challenges,
                &stage2.claims.batch_outputs,
                &stage2.sumcheck_proof,
                transcript,
                2,
            )
            .unwrap();
        sumchecks.append_output_claims(transcript, &stage2.claims.batch_outputs, &attached_product);
    }

    /// Stage 3's twin (`stage3::verify`'s clear body — the stage has no FR
    /// member): positions the transcript at the stage-4 boundary.
    pub(crate) fn replay_stage3<C: Clone + AppendToTranscript>(
        transcript: &mut Blake2bTranscript,
        stage1: &Stage1ProverOutput<Fr, C>,
        stage2: &Stage2ProverOutput<Fr, C>,
        stage3: &Stage3ProverOutput<Fr, C>,
    ) {
        let dimensions = TraceDimensions::new(LOG_T);
        let tau_low = stage2.clear_output.product_tau_low.clone();
        let product_remainder_point = stage2
            .clear_output
            .output_points
            .product_remainder_point()
            .to_vec();
        let sumchecks = Stage3Sumchecks {
            shift: SpartanShift::new(dimensions, tau_low.clone(), product_remainder_point.clone()),
            instruction_input: InstructionInput::new(dimensions, product_remainder_point),
            registers_claim_reduction: RegistersClaimReduction::new(dimensions, tau_low),
        };
        let challenges = sumchecks.draw_challenges(transcript).unwrap();
        sumchecks.validate_output_claims(&stage3.claims).unwrap();
        let input_values = stage3_input_values_from_upstream(
            &stage1.clear_output.output_values,
            &stage2.clear_output.output_values,
        );
        let input_points = sumchecks.empty_input_points();
        let _stage3_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &challenges,
                &stage3.claims,
                &stage3.sumcheck_proof,
                transcript,
                3,
            )
            .unwrap();
        sumchecks.append_output_claims(transcript, &stage3.claims);
    }

    /// Stage 4's twin (already round-tripped by stage 4's own test):
    /// `stage4::verify`'s clear body, positioning the transcript at the
    /// stage-5 boundary. The fixtures carry no advice and no committed
    /// program image, so the attached-claims step degenerates to the public
    /// initial-RAM evaluation alone.
    pub(crate) fn replay_stage4<C: Clone + AppendToTranscript>(
        transcript: &mut Blake2bTranscript,
        config: &ProverConfig,
        checked: &CheckedInputs,
        preprocessing: &FixturePreprocessing,
        stage2: &Stage2ProverOutput<Fr, C>,
        stage3: &Stage3ProverOutput<Fr, C>,
        stage4: &Stage4ProverOutput<Fr, C>,
    ) {
        let register_dimensions = config
            .rw_config
            .register_dimensions(LOG_T, REGISTER_ADDRESS_BITS);
        let ram_read_write_opening_point = stage2.clear_output.output_points.ram_read_write_point();
        let (r_address, _) = ram_read_write_opening_point.split_at(RAM_LOG_K);
        let public_eval =
            public_initial_ram_evaluation(checked, &preprocessing.verifier, r_address).unwrap();
        let init_structure =
            ram_val_check_init_structure(checked, false, r_address, public_eval).unwrap();
        let sumchecks = Stage4Sumchecks {
            registers_read_write: RegistersReadWriteChecking::new(register_dimensions),
            field_registers_read_write: stage4_field_inline::read_write_member(LOG_T),
            ram_val_check: RamValCheck::new(
                TraceDimensions::new(LOG_T),
                RAM_LOG_K,
                init_structure.decomposition(),
            ),
        };
        let challenges = sumchecks.draw_challenges(transcript).unwrap();
        sumchecks.validate_output_claims(&stage4.claims).unwrap();
        let ram_val_check_init = RamValCheckInitialEvaluation {
            public_eval,
            program_image_contribution: None,
            advice_contributions: Vec::new(),
        };
        let input_values = stage4_input_values_from_upstream(
            &stage2.clear_output.output_values,
            &stage3.clear_output.output_values,
            &ram_val_check_init,
        );
        let input_points = stage4_input_points_from_upstream(
            &stage2.clear_output.output_points,
            &stage3.clear_output.output_points,
            &init_structure,
        );
        let _stage4_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &challenges,
                &stage4.claims,
                &stage4.sumcheck_proof,
                transcript,
                4,
            )
            .unwrap();
        stage4.claims.append_to_transcript(transcript);
    }

    /// Stage 5's twin (`stage5::verify`'s clear body): positions the
    /// transcript at the stage-6a boundary.
    pub(crate) fn replay_stage5<C: Clone + AppendToTranscript>(
        transcript: &mut Blake2bTranscript,
        config: &ProverConfig,
        checked: &CheckedInputs,
        preprocessing: &FixturePreprocessing,
        stage2: &Stage2ProverOutput<Fr, C>,
        stage4: &Stage4ProverOutput<Fr, C>,
        stage5: &Stage5ProverOutput<Fr, C>,
    ) {
        let formula_dimensions = crate::stages::formula_dimensions(
            checked,
            config,
            preprocessing.verifier.program.bytecode_len(),
            JoltRelationId::InstructionReadRaf,
        )
        .unwrap();
        let trace_dimensions = formula_dimensions.trace;
        let sumchecks = Stage5Sumchecks {
            instruction_read_raf: InstructionReadRaf::new(formula_dimensions.instruction_read_raf),
            ram_ra_claim_reduction: RamRaClaimReduction::new(trace_dimensions, RAM_LOG_K),
            registers_val_evaluation: RegistersValEvaluation::new(trace_dimensions),
            field_registers_val_evaluation: stage5_field_inline::val_evaluation_member(
                trace_dimensions.log_t(),
            ),
        };
        let challenges = sumchecks.draw_challenges(transcript).unwrap();
        sumchecks.validate_output_claims(&stage5.claims).unwrap();
        let input_values = stage5_input_values_from_upstream(
            &stage2.clear_output.output_values,
            &stage4.clear_output.output_values,
        );
        let input_points = stage5_input_points_from_upstream(
            &stage2.clear_output.output_points,
            &stage4.clear_output.output_points,
        );
        let _stage5_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &challenges,
                &stage5.claims,
                &stage5.sumcheck_proof,
                transcript,
                5,
            )
            .unwrap();
        sumchecks.append_output_claims(transcript, &stage5.claims);
    }

    /// Stage 6a's twin (`stage6a::verify`'s clear body): positions the
    /// transcript at the stage-6b boundary.
    #[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
    pub(crate) fn replay_stage6a<C: Clone + AppendToTranscript>(
        transcript: &mut Blake2bTranscript,
        config: &ProverConfig,
        checked: &CheckedInputs,
        preprocessing: &FixturePreprocessing,
        stage1: &Stage1ProverOutput<Fr, C>,
        stage2: &Stage2ProverOutput<Fr, C>,
        stage3: &Stage3ProverOutput<Fr, C>,
        stage4: &Stage4ProverOutput<Fr, C>,
        stage5: &Stage5ProverOutput<Fr, C>,
        stage6a: &Stage6aProverOutput<Fr, C>,
    ) {
        let formula_dimensions = crate::stages::formula_dimensions(
            checked,
            config,
            preprocessing.verifier.program.bytecode_len(),
            JoltRelationId::BytecodeReadRaf,
        )
        .unwrap();
        let stage1_cycle_binding = stage1
            .clear_output
            .cycle_binding_checked(JoltRelationId::BytecodeReadRaf)
            .unwrap();
        let entry_bytecode_index = preprocessing
            .verifier
            .program
            .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)
            .unwrap();
        let sumchecks = Stage6aSumchecks::build_from_parts(Stage6aBuildParts {
            formula_dimensions: &formula_dimensions,
            committed_chunk_bits: config.one_hot_config.committed_chunk_bits(),
            committed_program: false,
            entry_bytecode_index,
            stage1_cycle_binding: &stage1_cycle_binding,
            stage2_points: &stage2.clear_output.output_points,
            stage3_points: &stage3.clear_output.output_points,
            stage4_points: &stage4.clear_output.output_points,
            stage5_points: &stage5.clear_output.output_points,
        })
        .unwrap();
        stage6a_field_inline::attach_bytecode_geometry(
            &sumchecks.bytecode_read_raf,
            stage6a_field_inline::preprocessed_bytecode_table(&preprocessing.verifier.program)
                .unwrap(),
            &stage4.clear_output.output_points,
            &stage5.clear_output.output_points,
        )
        .unwrap();
        let challenges = sumchecks.draw_challenges(transcript).unwrap();
        sumchecks.validate_output_claims(&stage6a.claims).unwrap();
        stage6a_field_inline::attach_bytecode_inputs(
            &sumchecks.bytecode_read_raf,
            &stage1.clear_output,
            &stage4.clear_output.output_values,
            &stage5.clear_output.output_values,
        )
        .unwrap();
        let base_input_values = bytecode_read_raf_address_phase_input_values_from_upstream(
            &stage1.clear_output.output_values,
            &stage2.clear_output.output_values,
            &stage3.clear_output.output_values,
            &stage4.clear_output.output_values,
            &stage5.clear_output.output_values,
        );
        // The packed shape folds the four reduced Inc claims into the
        // fused-inc consumer stage slots (stage6a::verify's own wrapper).
        #[cfg(feature = "akita")]
        let base_input_values = LatticeReadRafAddressPhaseInputClaims {
                base: base_input_values,
                inc: jolt_verifier::stages::stage6b::inc_claim_reduction::inc_claim_reduction_input_values_from_upstream(
                    &stage2.clear_output.output_values,
                    &stage4.clear_output.output_values,
                    &stage5.clear_output.output_values,
                ),
            };
        let input_values = Stage6aInputClaims {
            bytecode_read_raf: base_input_values,
            booleanity: BooleanityAddressPhaseInputClaims::default(),
        };
        let input_points = sumchecks.empty_input_points();
        let _stage6a_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &challenges,
                &stage6a.claims,
                &stage6a.sumcheck_proof,
                transcript,
                6,
            )
            .unwrap();
        sumchecks.append_output_claims(transcript, &stage6a.claims);
    }
}
