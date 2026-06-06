use jolt_prover_harness::{
    compare_named_values, registered_frontiers, CommitmentCheckpoint, ComparisonTarget,
    FeatureMode, FixtureArtifacts, FixtureKind, FixtureProvider, FixtureRequest, FixtureSource,
    FrontierCheckpoint, FrontierSpec, NamedValue, StaticFixtureProvider,
};

#[cfg(not(feature = "field-inline"))]
use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
use jolt_backends::{
    cpu::{CpuBackend, CpuBackendConfig},
    CommitmentBackend, CommitmentRequest, CommitmentRequestItem, CommitmentSlot,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::bytecode as verifier_field_bytecode;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig,
};
#[cfg(feature = "field-inline")]
use jolt_crypto::{Commitment, VectorCommitment};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{mock::MockCommitmentScheme, CommitmentScheme};
use jolt_poly::OneHotIndexOrder;
#[cfg(feature = "field-inline")]
use jolt_poly::Polynomial;
#[cfg(feature = "field-inline")]
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
use jolt_program::{
    execution::{JoltProgram, OwnedTrace, RegisterRead, RegisterState, TraceOutput, TraceRow},
    preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing},
};
#[cfg(feature = "field-inline")]
use jolt_prover::stages::stage0::CommitmentStageOutput;
use jolt_prover::stages::stage0::{prove, CommitmentStageConfig, CommitmentStageInput};
#[cfg(feature = "field-inline")]
use jolt_riscv::{FieldInlineOp, RV64IMAC_JOLT_FIELD_INLINE};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
#[cfg(feature = "field-inline")]
use jolt_sumcheck::{ClearProof, ClearSumcheckProof, SumcheckProof};
#[cfg(feature = "field-inline")]
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
#[cfg(feature = "field-inline")]
use jolt_verifier::proof::{JoltProof, JoltProofClaims, JoltStageProofs};
#[cfg(feature = "field-inline")]
use jolt_verifier::JoltVerifierPreprocessing;
use jolt_witness::protocols::jolt_vm::{
    JoltVmNamespace, JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
    RV64_LOOKUP_ADDRESS_BITS,
};
use jolt_witness::{
    MaterializationPolicy, OracleRef, PolynomialChunkKind, PolynomialEncoding, RetentionHint,
    ViewRequirement,
};

fn stage0_frontier() -> Result<FrontierSpec, String> {
    frontier_by_name("stage0_commitments")
}

fn frontier_by_name(name: &'static str) -> Result<FrontierSpec, String> {
    registered_frontiers()
        .map_err(|error| error.to_string())?
        .find(name)
        .copied()
        .ok_or_else(|| format!("registered {name} frontier is missing"))
}

fn kernel_evidence(
    kernel: &str,
) -> Result<Vec<jolt_prover_harness::KernelBenchmarkEvidence>, String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let port = ledger
        .find(kernel)
        .ok_or_else(|| format!("{kernel} ledger entry is missing"))?;
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| "failed to locate workspace root".to_owned())?;
    port.certification_evidence_files
        .iter()
        .map(|path| {
            jolt_prover_harness::KernelBenchmarkEvidence::read_json(&workspace_root.join(path))
                .map_err(|error| error.to_string())
        })
        .collect()
}

type MockPcs = MockCommitmentScheme<Fr>;

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct MockRoundCommitment(u64);

#[cfg(feature = "field-inline")]
impl AppendToTranscript for MockRoundCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append_bytes(&self.0.to_be_bytes());
    }
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct MockVectorCommitment;

#[cfg(feature = "field-inline")]
impl Commitment for MockVectorCommitment {
    type Output = MockRoundCommitment;
}

#[cfg(feature = "field-inline")]
impl VectorCommitment for MockVectorCommitment {
    type Field = Fr;
    type Setup = usize;

    fn capacity(setup: &Self::Setup) -> usize {
        *setup
    }

    fn commit(
        _setup: &Self::Setup,
        values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> Self::Output {
        MockRoundCommitment(values.len() as u64)
    }

    fn verify(
        _setup: &Self::Setup,
        _commitment: &Self::Output,
        _values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> bool {
        true
    }
}

const ENTRY: u64 = common::constants::RAM_START_ADDRESS;

#[test]
fn stage0_commitment_frontier_is_replacement_ready_with_certified_kernel_evidence(
) -> Result<(), String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let frontier = stage0_frontier()?;
    let evidence = kernel_evidence("cpu_streaming_commitments")?;

    jolt_prover_harness::validate_frontier_replacement_ready(frontier, &known, &ledger, &evidence)
        .map_err(|error| error.to_string())
}

#[cfg(feature = "zk")]
#[test]
fn stage0_zk_commitment_frontier_is_replacement_ready_with_certified_kernel_evidence(
) -> Result<(), String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let frontier = frontier_by_name("stage0_zk_commitments")?;
    let evidence = kernel_evidence("cpu_zk_streaming_commitments")?;

    jolt_prover_harness::validate_frontier_replacement_ready(frontier, &known, &ledger, &evidence)
        .map_err(|error| error.to_string())
}

#[test]
fn stage0_advice_frontier_is_replacement_ready_with_certified_kernel_evidence() -> Result<(), String>
{
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let frontier = frontier_by_name("stage0_advice_commitments")?;
    let evidence = kernel_evidence("cpu_advice_commitment_contexts")?;

    jolt_prover_harness::validate_frontier_replacement_ready(frontier, &known, &ledger, &evidence)
        .map_err(|error| error.to_string())
}

#[cfg(feature = "field-inline")]
#[test]
fn stage0_field_inline_frontier_is_replacement_ready_with_certified_kernel_evidence(
) -> Result<(), String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let frontier = frontier_by_name("stage0_field_inline_commitments")?;
    let evidence = kernel_evidence("cpu_field_inline_commitments")?;

    jolt_prover_harness::validate_frontier_replacement_ready(frontier, &known, &ledger, &evidence)
        .map_err(|error| error.to_string())
}

fn jolt_instruction(
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

#[cfg(feature = "field-inline")]
const FIELD_ENTRY: u64 = common::constants::RAM_START_ADDRESS;

#[cfg(feature = "field-inline")]
fn field_instruction(
    instruction_kind: JoltInstructionKind,
    offset: usize,
    rd: Option<u8>,
    rs1: Option<u8>,
    rs2: Option<u8>,
    imm: i128,
) -> JoltInstructionRow {
    jolt_instruction(instruction_kind, offset, rd, rs1, rs2, imm)
}

#[cfg(feature = "field-inline")]
fn field_encoded(value: u64) -> FieldEncodedValue {
    FieldEncodedValue::from_u64(value)
}

#[cfg(feature = "field-inline")]
fn field_row(instruction: JoltInstructionRow, data: FieldInlineTraceData) -> TraceRow {
    TraceRow {
        instruction,
        field_inline: Some(data.into()),
        ..TraceRow::default()
    }
}

#[cfg(feature = "field-inline")]
fn field_inline_fixture() -> (Vec<JoltInstructionRow>, Vec<TraceRow>) {
    let load_a = field_instruction(
        JoltInstructionKind::FIELD_LOAD_IMM,
        0,
        Some(1),
        None,
        None,
        13,
    );
    let row0 = field_row(
        load_a,
        FieldInlineTraceData {
            op: Some(FieldInlineOp::LoadImm),
            rd: Some(FieldRegisterWrite {
                register: 1,
                pre_value: field_encoded(0),
                post_value: field_encoded(13),
            }),
            ..FieldInlineTraceData::default()
        },
    );
    let load_b = field_instruction(
        JoltInstructionKind::FIELD_LOAD_IMM,
        1,
        Some(2),
        None,
        None,
        17,
    );
    let row1 = field_row(
        load_b,
        FieldInlineTraceData {
            op: Some(FieldInlineOp::LoadImm),
            rd: Some(FieldRegisterWrite {
                register: 2,
                pre_value: field_encoded(0),
                post_value: field_encoded(17),
            }),
            ..FieldInlineTraceData::default()
        },
    );
    let mul = field_instruction(
        JoltInstructionKind::FIELD_MUL,
        2,
        Some(3),
        Some(1),
        Some(2),
        0,
    );
    let row2 = field_row(
        mul,
        FieldInlineTraceData {
            op: Some(FieldInlineOp::Mul),
            rs1: Some(FieldRegisterRead {
                register: 1,
                value: field_encoded(13),
            }),
            rs2: Some(FieldRegisterRead {
                register: 2,
                value: field_encoded(17),
            }),
            rd: Some(FieldRegisterWrite {
                register: 3,
                pre_value: field_encoded(0),
                post_value: field_encoded(221),
            }),
            product: Some(field_encoded(221)),
            ..FieldInlineTraceData::default()
        },
    );

    (vec![load_a, load_b, mul], vec![row0, row1, row2])
}

#[cfg(feature = "field-inline")]
fn stage0_field_inline_output(
) -> Result<(CommitmentStageOutput<MockPcs>, JoltProgramPreprocessing), String> {
    let one_hot = JoltOneHotConfig {
        log_k_chunk: 4,
        lookups_ra_virtual_log_k_chunk: 16,
    };
    let witness_config = JoltVmWitnessConfig::new(2, 16, one_hot);
    let (bytecode, rows) = field_inline_fixture();
    let program = JoltProgram::from_parts_with_profile(
        Vec::new(),
        bytecode.clone(),
        Vec::new(),
        FIELD_ENTRY + 4,
        FIELD_ENTRY,
        RV64IMAC_JOLT_FIELD_INLINE,
    );
    let preprocessing = JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(
            bytecode,
            FIELD_ENTRY,
            RV64IMAC_JOLT_FIELD_INLINE,
        )
        .map_err(|error| error.to_string())?,
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 4,
    };
    let dimensions = JoltFormulaDimensions::try_from(one_hot.dimensions(
        witness_config.log_t,
        RV64_LOOKUP_ADDRESS_BITS,
        preprocessing.bytecode.code_size,
        witness_config.ram_k,
    ))
    .map_err(|error| error.to_string())?;
    let trace = TraceOutput::new(OwnedTrace::new(rows), Default::default(), None);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });
    let field_witness = witness
        .field_inline_witness()
        .map_err(|error| error.to_string())?;

    let output = prove::<Fr, _, _, MockPcs>(
        CommitmentStageInput::new(
            &witness,
            &(),
            CommitmentStageConfig::new(dimensions.ra_layout, false, false),
            jolt_verifier::JoltProtocolConfig::for_zk(false),
            &field_witness,
        ),
        &mut backend,
    )
    .map_err(|error| error.to_string())?;

    Ok((output, preprocessing))
}

#[cfg(feature = "field-inline")]
fn verifier_field_inline_bytecode_rows(
    instructions: &[JoltInstructionRow],
) -> Result<Vec<verifier_field_bytecode::FieldInlineBytecodeRow>, String> {
    instructions
        .iter()
        .map(verifier_field_inline_bytecode_row)
        .collect()
}

#[cfg(feature = "field-inline")]
fn verifier_field_inline_bytecode_row(
    instruction: &JoltInstructionRow,
) -> Result<verifier_field_bytecode::FieldInlineBytecodeRow, String> {
    let mut row = verifier_field_bytecode::FieldInlineBytecodeRow::default();
    match instruction.instruction_kind {
        JoltInstructionKind::NoOp => {}
        JoltInstructionKind::FIELD_LOAD_IMM => {
            row.operands.rd = instruction.operands.rd;
            row.flags.load_imm = true;
        }
        JoltInstructionKind::FIELD_MUL => {
            row.operands.rd = instruction.operands.rd;
            row.operands.rs1 = instruction.operands.rs1;
            row.operands.rs2 = instruction.operands.rs2;
            row.flags.mul = true;
        }
        other => {
            return Err(format!(
                "field-inline verifier replay fixture does not map {other:?}"
            ));
        }
    }
    Ok(row)
}

#[cfg(feature = "field-inline")]
fn field_inline_verifier_stage_proofs() -> JoltStageProofs<Fr, MockVectorCommitment> {
    let proof = SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()));
    JoltStageProofs {
        stage1_uni_skip_first_round_proof: proof.clone(),
        stage1_sumcheck_proof: proof.clone(),
        stage2_uni_skip_first_round_proof: proof.clone(),
        stage2_sumcheck_proof: proof.clone(),
        stage3_sumcheck_proof: proof.clone(),
        stage4_sumcheck_proof: proof.clone(),
        stage5_sumcheck_proof: proof.clone(),
        stage6_sumcheck_proof: proof.clone(),
        stage7_sumcheck_proof: proof,
    }
}

#[cfg(feature = "field-inline")]
fn mock_joint_opening_proof() -> <MockPcs as CommitmentScheme>::Proof {
    let poly = Polynomial::new(vec![Fr::from_u64(0)]);
    let mut transcript = Blake2bTranscript::new(b"stage0-fi-opening");
    MockPcs::open(&poly, &[], Fr::from_u64(0), &(), None, &mut transcript)
}

#[test]
fn stage0_frontier_requires_correctness_and_performance_gates() -> Result<(), String> {
    let frontier = stage0_frontier()?;

    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
fn commitment_checkpoint_compares_by_logical_name() {
    let expected = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![
            NamedValue::new("RdInc", "c_rd"),
            NamedValue::new("RamInc", "c_ram"),
        ],
        opening_hints: vec![NamedValue::new("RdIncHint", "h_rd")],
    });
    let actual = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![
            NamedValue::new("RamInc", "c_ram"),
            NamedValue::new("RdInc", "c_rd"),
        ],
        opening_hints: vec![NamedValue::new("RdIncHint", "h_rd")],
    });

    let report = compare_named_values(
        ComparisonTarget::CoreCommitments,
        &expected.named_values(),
        &actual.named_values(),
    );
    assert!(report.is_success());
}

#[test]
fn commitment_checkpoint_rejects_duplicate_logical_names() {
    let expected = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![NamedValue::new("RdInc", "c_rd")],
        opening_hints: Vec::new(),
    });
    let actual = FrontierCheckpoint::Commitments(CommitmentCheckpoint {
        commitments: vec![
            NamedValue::new("RdInc", "c_rd"),
            NamedValue::new("RdInc", "c_rd_again"),
        ],
        opening_hints: Vec::new(),
    });

    let report = compare_named_values(
        ComparisonTarget::CoreCommitments,
        &expected.named_values(),
        &actual.named_values(),
    );
    assert!(!report.is_success());
    assert_eq!(report.mismatches[0].path, "actual.RdInc");
}

#[test]
fn fixture_provider_returns_typed_artifacts() -> Result<(), String> {
    let provider = StaticFixtureProvider;
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let artifacts = provider.load(&request).map_err(|error| error.to_string())?;

    assert_eq!(
        artifacts,
        FixtureArtifacts::new(
            FixtureKind::MuldivSmall,
            FeatureMode::Transparent,
            FixtureSource::ModularSynthetic,
        )
        .with_note("static harness typed artifact; not valid for frontier acceptance")
    );
    Ok(())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage0_cpu_commitment_verifier_replay_verifies_against_core_muldiv_fixture() -> Result<(), String>
{
    let frontier = stage0_frontier()?;
    assert!(frontier.requires_verifier_correctness());

    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = jolt_prover_harness::load_stage0_commitment_verifier_replay_fixture(&request)
        .map_err(|error| error.to_string())?;

    fixture.verify().map_err(|error| error.to_string())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage0_cpu_commitment_verifier_replay_verifies_against_core_advice_fixture() -> Result<(), String>
{
    let request = FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent);
    let fixture = jolt_prover_harness::load_stage0_commitment_verifier_replay_fixture(&request)
        .map_err(|error| error.to_string())?;

    assert!(fixture.trusted_advice_commitment.is_some());
    assert!(fixture.proof.untrusted_advice_commitment.is_some());
    fixture.verify().map_err(|error| error.to_string())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage0_advice_commitment_kernel_matches_core_contexts() -> Result<(), String> {
    let request = FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent);
    let fixture =
        jolt_prover_harness::load_stage0_advice_commitment_kernel_benchmark_fixture(&request)
            .map_err(|error| error.to_string())?;

    fixture
        .verify_commitment_parity()
        .map_err(|error| error.to_string())
}

#[test]
fn cpu_backend_streams_real_instruction_ra_as_one_hot_boundary() -> Result<(), String> {
    let one_hot = JoltOneHotConfig {
        log_k_chunk: 4,
        lookups_ra_virtual_log_k_chunk: 16,
    };
    let witness_config = JoltVmWitnessConfig::new(2, 16, one_hot);
    let instruction = jolt_instruction(JoltInstructionKind::ADDI, 0, Some(1), Some(2), None, -1);
    let preprocessing = JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing::preprocess(vec![instruction], ENTRY, RV64IMAC_JOLT)
            .map_err(|error| error.to_string())?,
        ram: RAMPreprocessing::default(),
        memory_layout: Default::default(),
        max_padded_trace_length: 4,
    };
    let trace = TraceOutput::new(
        OwnedTrace::new(vec![TraceRow {
            instruction,
            registers: RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 10,
                }),
                ..RegisterState::default()
            },
            ..TraceRow::default()
        }]),
        Default::default(),
        None,
    );
    let program =
        JoltProgram::from_parts(Vec::new(), vec![instruction], Vec::new(), ENTRY + 4, ENTRY);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(7),
        ViewRequirement::new(
            OracleRef::committed(JoltCommittedPolynomial::InstructionRa(15)),
            PolynomialEncoding::OneHot,
            MaterializationPolicy::Streaming,
            RetentionHint::ThroughStage8,
        ),
    )]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result = <CpuBackend as CommitmentBackend<Fr, JoltVmNamespace, MockPcs>>::commit(
        &mut backend,
        &request,
        &witness,
        &(),
    )
    .map_err(|error| error.to_string())?;

    assert_eq!(result.streamed_witness.len(), 1);
    assert_eq!(result.streamed_witness[0].slot, CommitmentSlot(7));
    assert_eq!(result.streamed_witness[0].rows, 4);
    assert_eq!(
        result.streamed_witness[0]
            .chunks
            .iter()
            .map(|chunk| (chunk.kind, chunk.rows))
            .collect::<Vec<_>>(),
        vec![
            (PolynomialChunkKind::OneHot, 2),
            (PolynomialChunkKind::OneHot, 2),
        ]
    );

    let expected_poly = jolt_poly::OneHotPolynomial::new_with_index_order(
        16,
        vec![Some(1), Some(0), Some(0), Some(0)],
        OneHotIndexOrder::ColumnMajor,
    );
    let (expected_commitment, ()) = MockPcs::commit(&expected_poly, &());
    assert_eq!(result.commitments.len(), 1);
    assert_eq!(result.commitments[0].slot, CommitmentSlot(7));
    assert_eq!(
        result.commitments[0].oracle,
        OracleRef::committed(JoltCommittedPolynomial::InstructionRa(15))
    );
    assert_eq!(result.commitments[0].rows, 64);
    assert_eq!(result.commitments[0].commitment, expected_commitment);
    Ok(())
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn stage0_cpu_frontier_commits_advice_from_normalized_jolt_vm_witness() -> Result<(), String> {
    let one_hot = JoltOneHotConfig {
        log_k_chunk: 4,
        lookups_ra_virtual_log_k_chunk: 16,
    };
    let witness_config = JoltVmWitnessConfig::new(1, 16, one_hot)
        .include_trusted_advice(true)
        .include_untrusted_advice(true);
    let memory_config = MemoryConfig {
        max_input_size: 0,
        max_trusted_advice_size: 64,
        max_untrusted_advice_size: 64,
        max_output_size: 0,
        stack_size: 0,
        heap_size: 0,
        program_size: Some(64),
    };
    let memory_layout = MemoryLayout::new(&memory_config);
    let preprocessing = JoltProgramPreprocessing {
        bytecode: BytecodePreprocessing {
            code_size: 2,
            ..Default::default()
        },
        ram: RAMPreprocessing::default(),
        memory_layout: memory_layout.clone(),
        max_padded_trace_length: 2,
    };
    let dimensions = JoltFormulaDimensions::try_from(one_hot.dimensions(
        witness_config.log_t,
        RV64_LOOKUP_ADDRESS_BITS,
        preprocessing.bytecode.code_size,
        witness_config.ram_k,
    ))
    .map_err(|error| error.to_string())?;
    let device = JoltDevice {
        trusted_advice: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        untrusted_advice: vec![0xaa, 0xbb],
        memory_layout,
        ..Default::default()
    };
    let program = JoltProgram::default();
    let trace = TraceOutput::new(OwnedTrace::default(), device, None);
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&program, &preprocessing, trace),
    );
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 3,
    });

    let output = prove::<Fr, _, _, MockPcs>(
        CommitmentStageInput::new(
            &witness,
            &(),
            CommitmentStageConfig::new(dimensions.ra_layout, true, true),
            jolt_verifier::JoltProtocolConfig::for_zk(false),
        ),
        &mut backend,
    )
    .map_err(|error| error.to_string())?;

    let trusted_poly = jolt_poly::Polynomial::new(vec![
        Fr::from_u64(0x0807_0605_0403_0201),
        Fr::from_u64(0x0a09),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
    ]);
    let untrusted_poly = jolt_poly::Polynomial::new(vec![
        Fr::from_u64(0xbbaa),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
        Fr::from_u64(0),
    ]);
    let (trusted_commitment, ()) = MockPcs::commit(&trusted_poly, &());
    let (untrusted_commitment, ()) = MockPcs::commit(&untrusted_poly, &());

    assert_eq!(
        output.trusted_advice_commitment.as_ref(),
        Some(&trusted_commitment)
    );
    assert_eq!(
        output.untrusted_advice_commitment.as_ref(),
        Some(&untrusted_commitment)
    );
    Ok(())
}

#[cfg(feature = "field-inline")]
#[test]
fn stage0_cpu_frontier_commits_field_inline_namespace_with_jolt_vm_witness() -> Result<(), String> {
    let (output, _) = stage0_field_inline_output()?;

    let field_poly = jolt_poly::Polynomial::new(vec![
        Fr::from_u64(13),
        Fr::from_u64(17),
        Fr::from_u64(221),
        Fr::from_u64(0),
    ]);
    let (field_commitment, ()) = MockPcs::commit(&field_poly, &());

    assert_eq!(
        output.commitments.field_inline.field_registers.rd_inc,
        field_commitment
    );
    assert!(output
        .prover_state
        .field_inline_opening_hints
        .contains_key(&FieldInlineCommittedPolynomial::FieldRdInc));
    Ok(())
}

#[cfg(feature = "field-inline")]
#[test]
fn stage0_field_inline_commitments_are_accepted_by_jolt_verifier_replay() -> Result<(), String> {
    let (output, preprocessing) = stage0_field_inline_output()?;
    let verifier_bytecode = verifier_field_inline_bytecode_rows(&preprocessing.bytecode.bytecode)?;
    let expected_field_inline_transcript = verifier_field_bytecode::bytecode_transcript_bytes(
        &verifier_bytecode,
        jolt_verifier::config::SELECTED_FIELD_INLINE_CONFIG.field_register_log_k,
    );
    let verifier_preprocessing = JoltVerifierPreprocessing::<MockPcs, MockVectorCommitment>::new(
        preprocessing.clone(),
        [3; 32],
        (),
        None,
    )
    .with_field_inline_bytecode(verifier_bytecode);
    let public_io = common::jolt_device::JoltDevice {
        memory_layout: preprocessing.memory_layout.clone(),
        ..Default::default()
    };
    let proof = JoltProof::<MockPcs, MockVectorCommitment>::new(
        output.commitments,
        field_inline_verifier_stage_proofs(),
        mock_joint_opening_proof(),
        output.untrusted_advice_commitment,
        JoltProofClaims::Clear(jolt_verifier::compat::claims::empty_clear_opening_claims(4)),
        4,
        16,
        jolt_claims::protocols::jolt::JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: 1,
            ram_rw_phase2_num_rounds: 1,
            registers_rw_phase1_num_rounds: 1,
            registers_rw_phase2_num_rounds: 1,
        },
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
        jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
    );
    let verifier_state = jolt_verifier::verify_until_stage1::<
        MockPcs,
        MockVectorCommitment,
        Blake2bTranscript,
        _,
    >(&verifier_preprocessing, &public_io, &proof, None, false)
    .map_err(|error| error.to_string())?;

    assert_eq!(
        verifier_state.checked.field_inline_bytecode_transcript,
        expected_field_inline_transcript
    );
    Ok(())
}
