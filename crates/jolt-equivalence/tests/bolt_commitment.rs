//! Commitment-phase transcript bridge between Bolt IR and jolt-core.
//!
//! This keeps the first oracle intentionally narrow: Bolt owns the commitment
//! ordering through its CPU IR, while jolt-core owns the reference
//! `append_serializable` transcript semantics for the same Dory commitments.

use std::collections::BTreeMap;

use ark_serialize::CanonicalSerialize;
use common::constants::{RAM_START_ADDRESS, XLEN};
use common::jolt_device::JoltDevice;
use jolt_compiler_v2::{
    build_commitment_protocol, build_stage1_outer_protocol, build_stage2_protocol,
    commitment_cpu_program, lower_commitment_to_compute, lower_compute_to_cpu,
    lower_piop_and_fiat_shamir, lower_stage1_to_compute, lower_stage2_to_compute,
    project_prover_party, project_verifier_party, resolve_compute_kernels, stage1_cpu_program,
    stage2_cpu_program, CommitmentCpuProgram, JoltProtocolParams, MeliorContext,
    OptionalSkipPolicy, OracleGeneration, Role, TranscriptStep,
};
use jolt_compiler_v2::{
    Stage1CpuProgram as CompilerStage1CpuProgram, Stage2CpuProgram as CompilerStage2CpuProgram,
};
use jolt_core::curve::Bn254Curve;
use jolt_core::host;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme as CoreCommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::opening_proof::{OpeningId, SumcheckId};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::univariate_skip::{
    UniSkipFirstRoundProof, UniSkipFirstRoundProofVariant,
};
use jolt_core::transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _};
use jolt_core::zkvm::instruction::{CircuitFlags, InstructionFlags, LookupQuery};
use jolt_core::zkvm::proof_serialization::{Claims as CoreClaims, JoltProof as CoreJoltProof};
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::r1cs::inputs::ProductCycleInputs as CoreProductCycleInputs;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::spartan::verify_stage2_uni_skip;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};
use jolt_equivalence::checkpoint::{
    assert_transcripts_match, CheckpointTranscript, TranscriptEvent,
};
use jolt_equivalence::cross_verifier::conversion::{
    commitment_to_ark, to_ark, to_core_sumcheck_proof,
};
use jolt_field::{Field, Fr};
use jolt_host::{extract_trace, BytecodePreprocessing, Program};
use jolt_kernels::stage1::{
    execute_stage1_program, Stage1CpuProgramPlan as KernelStage1CpuProgramPlan,
    Stage1ExecutionArtifacts, Stage1ExecutionMode, Stage1KernelError,
    Stage1KernelPlan as KernelStage1KernelPlan,
    Stage1OpeningBatchPlan as KernelStage1OpeningBatchPlan,
    Stage1OpeningClaimPlan as KernelStage1OpeningClaimPlan, Stage1OuterR1csData, Stage1Params,
    Stage1Proof, Stage1ProverInputs, Stage1ProverKernelExecutor,
    Stage1SumcheckBatchPlan as KernelStage1SumcheckBatchPlan,
    Stage1SumcheckClaimPlan as KernelStage1SumcheckClaimPlan,
    Stage1SumcheckDriverPlan as KernelStage1SumcheckDriverPlan,
    Stage1SumcheckEvalPlan as KernelStage1SumcheckEvalPlan,
    Stage1SumcheckInstanceResultPlan as KernelStage1SumcheckInstanceResultPlan,
    Stage1TranscriptSqueezePlan, Stage1VerifierKernelExecutor,
};
use jolt_kernels::stage2::{
    execute_stage2_program, product_virtual_uniskip_extended_evals,
    Stage2ChallengeExtractPlan as KernelStage2ChallengeExtractPlan,
    Stage2CpuProgramPlan as KernelStage2CpuProgramPlan, Stage2ExecutionArtifacts,
    Stage2ExecutionMode, Stage2FieldConstantPlan as KernelStage2FieldConstantPlan,
    Stage2FieldExprPlan as KernelStage2FieldExprPlan, Stage2InstructionLookupCycle,
    Stage2KernelPlan as KernelStage2KernelPlan, Stage2OpeningBatchPlan, Stage2OpeningClaimPlan,
    Stage2OpeningInputPlan, Stage2OpeningInputValue, Stage2Params, Stage2PointConcatPlan,
    Stage2PointSlicePlan, Stage2ProductVirtualCycle,
    Stage2ProgramStepPlan as KernelStage2ProgramStepPlan, Stage2Proof, Stage2ProverInputs,
    Stage2ProverKernelExecutor, Stage2RamAccess, Stage2RamData, Stage2RamOutputLayout,
    Stage2SumcheckBatchPlan as KernelStage2SumcheckBatchPlan,
    Stage2SumcheckClaimPlan as KernelStage2SumcheckClaimPlan,
    Stage2SumcheckDriverPlan as KernelStage2SumcheckDriverPlan,
    Stage2SumcheckEvalPlan as KernelStage2SumcheckEvalPlan, Stage2SumcheckInstanceResultPlan,
    Stage2TranscriptSqueezePlan as KernelStage2TranscriptSqueezePlan, Stage2VerifierKernelExecutor,
};
use jolt_openings::StreamingCommitment;
use jolt_poly::UnivariatePoly;
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use jolt_verifier::TRANSCRIPT_LABEL;
use jolt_witness::CycleInput;
use jolt_witness_v2::{
    dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle,
};
use tracer::instruction::RAMAccess;

type CoreFr = ark_bn254::Fr;
type CoreCommitment = <DoryCommitmentScheme as CoreCommitmentScheme>::Commitment;
type CoreProver<'a> =
    JoltCpuProver<'a, CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreProof = CoreJoltProof<CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreVerifier<'a> =
    JoltVerifier<'a, CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreVerifierPreprocessing =
    JoltVerifierPreprocessing<CoreFr, Bn254Curve, DoryCommitmentScheme>;

#[test]
fn bolt_commitment_transcript_matches_jolt_core_append_serializable() {
    let (prover_program, verifier_program) = bolt_commitment_programs();
    let prover_trace = run_bolt_commitment_prover(&prover_program);
    let verifier_trace = run_bolt_commitment_verifier(&verifier_program, &prover_trace.commitments);
    let core_log = core_commitment_log(&prover_trace, &prover_program.transcript_steps);

    assert_eq!(prover_trace.commitments, verifier_trace.commitments);
    assert_transcripts_match(&prover_trace.log, &verifier_trace.log);
    assert_transcripts_match(&core_log, &prover_trace.log);
}

#[test]
fn bolt_commitment_real_muldiv_trace_matches_jolt_core() {
    let fixture = core_muldiv_commitment_fixture();
    let (prover_program, verifier_program) = bolt_commitment_programs_with_params(&fixture.params);
    let oracle_data = real_muldiv_oracle_data(&prover_program, &fixture.cycle_inputs);

    let prover_trace = run_bolt_commitment_prover_with(
        &prover_program,
        &fixture.pcs_setup,
        |oracle, _num_vars| oracle_data.get(oracle).cloned().flatten(),
    );
    let verifier_trace = run_bolt_commitment_verifier(&verifier_program, &prover_trace.commitments);

    let bolt_core_commitments = prover_trace
        .commitments
        .iter()
        .filter_map(|commitment| commitment.as_ref())
        .take(fixture.commitments.len())
        .map(commitment_to_ark)
        .collect::<Vec<_>>();
    assert_eq!(bolt_core_commitments, fixture.commitments);

    let core_log =
        core_commitments_transcript_log(&fixture.commitments, &prover_program.transcript_steps);
    assert_eq!(prover_trace.commitments, verifier_trace.commitments);
    assert_transcripts_match(&prover_trace.log, &verifier_trace.log);
    assert_transcripts_match(&core_log, &prover_trace.log);
}

#[test]
fn bolt_commitment_stage1_real_muldiv_parity_checks() {
    let fixture = core_muldiv_commitment_fixture();
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, stage1_verifier_program) =
        bolt_stage1_programs_with_params(&fixture.params);
    let oracle_data = real_muldiv_oracle_data(&commitment_prover_program, &fixture.cycle_inputs);

    let commitment_prover_trace = run_bolt_commitment_prover_with(
        &commitment_prover_program,
        &fixture.pcs_setup,
        |oracle, _num_vars| oracle_data.get(oracle).cloned().flatten(),
    );
    let commitment_verifier_trace = run_bolt_commitment_verifier(
        &commitment_verifier_program,
        &commitment_prover_trace.commitments,
    );

    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let stage1_verifier_plan = leak_stage1_program(&stage1_verifier_program);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterR1csData::new(&r1cs_key, &fixture.r1cs_witness)
        .expect("valid R1CS witness shape");

    let mut prover_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(
        &mut prover_transcript,
        &fixture.io,
        fixture.proof.ram_K,
        fixture.proof.trace_length,
        fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        &mut prover_transcript,
        &commitment_prover_trace.records,
        &commitment_prover_trace.commitments,
        &commitment_prover_program.transcript_steps,
    );
    let stage1_inputs =
        Stage1ProverInputs::empty(r1cs_key.num_cycle_vars()).with_outer_remaining_evaluator(&data);
    let mut stage1_prover = Stage1ProverKernelExecutor::new(stage1_inputs);
    let stage1_artifacts = execute_stage1_program(
        stage1_prover_plan,
        Stage1ExecutionMode::Prover,
        &mut stage1_prover,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let mut verifier_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(
        &mut verifier_transcript,
        &fixture.io,
        fixture.proof.ram_K,
        fixture.proof.trace_length,
        fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        &mut verifier_transcript,
        &commitment_verifier_trace.records,
        &commitment_verifier_trace.commitments,
        &commitment_verifier_program.transcript_steps,
    );
    let mut stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
    let verified_stage1 = execute_stage1_program(
        stage1_verifier_plan,
        Stage1ExecutionMode::Verifier,
        &mut stage1_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts prover proof");

    assert_eq!(
        stage1_artifacts.sumchecks.len(),
        verified_stage1.sumchecks.len()
    );
    assert_transcripts_match(prover_transcript.log(), verifier_transcript.log());

    assert_core_accepts_bolt_stage1(&fixture, &stage1_artifacts);
    assert_core_states_match_bolt_stage1(&fixture, prover_transcript.log());
    assert_bolt_stage1_tamper_rejected(
        stage1_verifier_plan,
        &stage1_proof,
        &fixture,
        &commitment_verifier_trace,
        &commitment_verifier_program.transcript_steps,
    );
}

#[test]
fn bolt_stage2_product_uniskip_real_muldiv_matches_jolt_core() {
    let fixture = core_muldiv_commitment_fixture();
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, _) = bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, _) = bolt_stage2_programs_with_params(&fixture.params);
    let oracle_data = real_muldiv_oracle_data(&commitment_prover_program, &fixture.cycle_inputs);

    let commitment_prover_trace = run_bolt_commitment_prover_with(
        &commitment_prover_program,
        &fixture.pcs_setup,
        |oracle, _num_vars| oracle_data.get(oracle).cloned().flatten(),
    );
    let commitment_verifier_trace = run_bolt_commitment_verifier(
        &commitment_verifier_program,
        &commitment_prover_trace.commitments,
    );

    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterR1csData::new(&r1cs_key, &fixture.r1cs_witness)
        .expect("valid R1CS witness shape");

    let mut bolt_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(
        &mut bolt_transcript,
        &fixture.io,
        fixture.proof.ram_K,
        fixture.proof.trace_length,
        fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        &mut bolt_transcript,
        &commitment_prover_trace.records,
        &commitment_prover_trace.commitments,
        &commitment_prover_program.transcript_steps,
    );
    let stage1_inputs =
        Stage1ProverInputs::empty(r1cs_key.num_cycle_vars()).with_outer_remaining_evaluator(&data);
    let mut stage1_prover = Stage1ProverKernelExecutor::new(stage1_inputs);
    let stage1_artifacts = execute_stage1_program(
        stage1_prover_plan,
        Stage1ExecutionMode::Prover,
        &mut stage1_prover,
        &mut bolt_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");

    let stage2_openings = stage2_product_opening_inputs(&stage1_artifacts);
    let tau_low = &stage2_openings[0].point;
    let extended_evals =
        product_virtual_uniskip_extended_evals(&fixture.product_virtual_cycles, tau_low)
            .expect("product virtual extended evals");
    let stage2_inputs = Stage2ProverInputs::new(&stage2_openings)
        .with_product_uniskip_extended_evals(&extended_evals);
    let stage2_prover_plan = leak_stage2_product_uniskip_program(&stage2_prover_program);
    let mut stage2_prover = Stage2ProverKernelExecutor::new(stage2_inputs);
    let stage2_artifacts = execute_stage2_program(
        stage2_prover_plan,
        Stage2ExecutionMode::Prover,
        &mut stage2_prover,
        &mut bolt_transcript,
    )
    .expect("Bolt Stage 2 product uni-skip succeeds");

    assert_eq!(stage2_artifacts.sumchecks.len(), 1);
    assert_core_stage2_uniskip_proof_matches_bolt(&fixture.proof, &stage2_artifacts.sumchecks[0]);
    assert_bolt_chain_verifier_accepts_stage2_product_uniskip(BoltStage2ChainVerifierInput {
        fixture: &fixture,
        commitment_verifier_program: &commitment_verifier_program,
        commitment_verifier_trace: &commitment_verifier_trace,
        stage1_prover_plan,
        stage2_prover_plan,
        stage1_artifacts: &stage1_artifacts,
        stage2_artifacts: &stage2_artifacts,
        prover_transcript: &bolt_transcript,
    });

    let mut core_verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        clone_core_proof(&fixture.proof),
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    core_verifier.run_preamble();
    let _ = core_verifier
        .verify_stage1()
        .expect("core Stage 1 verifies before Stage 2");
    let stage2_uniskip_proof = core_verifier
        .proof
        .stage2_uni_skip_first_round_proof
        .clone();
    let _ = verify_stage2_uni_skip(
        &stage2_uniskip_proof,
        &mut core_verifier.opening_accumulator,
        &mut core_verifier.transcript,
    )
    .expect("core Stage 2 uni-skip verifies");

    let core_states = core_verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_transcript.log());
    assert_state_history_match(&core_states, &bolt_states);

    assert_eq!(
        commitment_prover_trace.commitments,
        commitment_verifier_trace.commitments
    );
}

#[test]
fn bolt_stage2_batched_real_muldiv_self_parity() {
    let fixture = core_muldiv_commitment_fixture();
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, _) = bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, _) = bolt_stage2_programs_with_params(&fixture.params);
    let oracle_data = real_muldiv_oracle_data(&commitment_prover_program, &fixture.cycle_inputs);

    let commitment_prover_trace = run_bolt_commitment_prover_with(
        &commitment_prover_program,
        &fixture.pcs_setup,
        |oracle, _num_vars| oracle_data.get(oracle).cloned().flatten(),
    );
    let commitment_verifier_trace = run_bolt_commitment_verifier(
        &commitment_verifier_program,
        &commitment_prover_trace.commitments,
    );

    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterR1csData::new(&r1cs_key, &fixture.r1cs_witness)
        .expect("valid R1CS witness shape");

    let mut prover_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(
        &mut prover_transcript,
        &fixture.io,
        fixture.proof.ram_K,
        fixture.proof.trace_length,
        fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        &mut prover_transcript,
        &commitment_prover_trace.records,
        &commitment_prover_trace.commitments,
        &commitment_prover_program.transcript_steps,
    );
    let stage1_inputs =
        Stage1ProverInputs::empty(r1cs_key.num_cycle_vars()).with_outer_remaining_evaluator(&data);
    let mut stage1_prover = Stage1ProverKernelExecutor::new(stage1_inputs);
    let stage1_artifacts = execute_stage1_program(
        stage1_prover_plan,
        Stage1ExecutionMode::Prover,
        &mut stage1_prover,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");

    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let tau_low = stage2_openings
        .iter()
        .find(|input| input.symbol == "stage2.input.stage1.Product")
        .expect("product opening")
        .point
        .as_slice();
    let extended_evals =
        product_virtual_uniskip_extended_evals(&fixture.product_virtual_cycles, tau_low)
            .expect("product virtual extended evals");
    let ram_data = Stage2RamData {
        log_k: fixture.params.log_k_ram,
        start_address: fixture.ram_start_address,
        initial_ram: &fixture.initial_ram_state,
        final_ram: &fixture.final_ram_state,
        accesses: &fixture.ram_accesses,
        output_layout: Some(fixture.ram_output_layout),
    };
    let stage2_inputs = Stage2ProverInputs::new(&stage2_openings)
        .with_product_uniskip_extended_evals(&extended_evals)
        .with_product_virtual_cycles(&fixture.product_virtual_cycles)
        .with_instruction_lookup_cycles(&fixture.instruction_lookup_cycles)
        .with_ram_data(&ram_data);
    let stage2_prover_plan = leak_stage2_program(&stage2_prover_program);
    let mut stage2_prover = Stage2ProverKernelExecutor::new(stage2_inputs);
    let stage2_artifacts = execute_stage2_program(
        stage2_prover_plan,
        Stage2ExecutionMode::Prover,
        &mut stage2_prover,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 2 prover succeeds");

    let mut verifier_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(
        &mut verifier_transcript,
        &fixture.io,
        fixture.proof.ram_K,
        fixture.proof.trace_length,
        fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        &mut verifier_transcript,
        &commitment_verifier_trace.records,
        &commitment_verifier_trace.commitments,
        &commitment_verifier_program.transcript_steps,
    );
    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let mut stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
    let verified_stage1 = execute_stage1_program(
        stage1_prover_plan,
        Stage1ExecutionMode::Verifier,
        &mut stage1_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts");
    let stage2_verifier_openings = stage2_opening_inputs(&verified_stage1);
    let stage2_proof = Stage2Proof::from(stage2_artifacts.clone());
    let mut stage2_verifier =
        Stage2VerifierKernelExecutor::new(&stage2_proof, &stage2_verifier_openings)
            .with_ram_data(&ram_data);
    let verified_stage2 = execute_stage2_program(
        stage2_prover_plan,
        Stage2ExecutionMode::Verifier,
        &mut stage2_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts");

    assert_eq!(
        stage2_artifacts.sumchecks.len(),
        verified_stage2.sumchecks.len()
    );
    assert_state_history_match(
        &transcript_states(prover_transcript.log()),
        &transcript_states(verifier_transcript.log()),
    );
    assert_core_accepts_bolt_stage2(&fixture, &stage1_artifacts, &stage2_artifacts);
    assert_core_states_match_bolt_stage2(&fixture, prover_transcript.log());
}

#[derive(Clone, Debug)]
struct CommitmentRecord {
    artifact: String,
}

#[derive(Clone, Debug)]
struct BoltCommitmentTrace {
    commitments: Vec<Option<DoryCommitment>>,
    records: Vec<CommitmentRecord>,
    log: Vec<TranscriptEvent>,
}

struct CoreMuldivCommitmentFixture {
    params: JoltProtocolParams,
    pcs_setup: DoryProverSetup,
    proof: CoreProof,
    verifier_preprocessing: &'static CoreVerifierPreprocessing,
    io: JoltDevice,
    entry_address: u64,
    cycle_inputs: Vec<CycleInput>,
    r1cs_witness: Vec<Fr>,
    product_virtual_cycles: Vec<Stage2ProductVirtualCycle>,
    instruction_lookup_cycles: Vec<Stage2InstructionLookupCycle>,
    ram_accesses: Vec<Stage2RamAccess>,
    initial_ram_state: Vec<u64>,
    final_ram_state: Vec<u64>,
    ram_start_address: u64,
    ram_output_layout: Stage2RamOutputLayout,
    commitments: Vec<CoreCommitment>,
}

fn bolt_commitment_programs() -> (CommitmentCpuProgram, CommitmentCpuProgram) {
    let params = JoltProtocolParams::new(0, 0, 0);
    bolt_commitment_programs_with_params(&params)
}

fn bolt_commitment_programs_with_params(
    params: &JoltProtocolParams,
) -> (CommitmentCpuProgram, CommitmentCpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_commitment_protocol(&context, params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_commitment_to_compute(&context, &prover_party).expect("lower prover compute");
    let verifier_compute =
        lower_commitment_to_compute(&context, &verifier_party).expect("lower verifier compute");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = commitment_cpu_program(&prover_cpu).expect("extract prover CPU program");
    let verifier_program =
        commitment_cpu_program(&verifier_cpu).expect("extract verifier CPU program");
    (prover_program, verifier_program)
}

fn bolt_stage1_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage1CpuProgram, CompilerStage1CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage1_outer_protocol(&context, params).expect("build stage1 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 1 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage1_to_compute(&context, &prover_party).expect("lower prover Stage 1");
    let verifier_compute =
        lower_stage1_to_compute(&context, &verifier_party).expect("lower verifier Stage 1");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage1_cpu_program(&prover_cpu).expect("extract prover Stage 1 CPU");
    let verifier_program = stage1_cpu_program(&verifier_cpu).expect("extract verifier Stage 1 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage2_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage2CpuProgram, CompilerStage2CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage2_protocol(&context, params).expect("build stage2 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 2 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage2_to_compute(&context, &prover_party).expect("lower prover Stage 2");
    let verifier_compute =
        lower_stage2_to_compute(&context, &verifier_party).expect("lower verifier Stage 2");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage2_cpu_program(&prover_cpu).expect("extract prover Stage 2 CPU");
    let verifier_program = stage2_cpu_program(&verifier_cpu).expect("extract verifier Stage 2 CPU");
    (prover_program, verifier_program)
}

fn leak_stage1_program(program: &CompilerStage1CpuProgram) -> &'static KernelStage1CpuProgramPlan {
    let transcript_squeezes = leak_slice(
        program
            .transcript_squeezes
            .iter()
            .map(|plan| Stage1TranscriptSqueezePlan {
                symbol: leak_str(&plan.symbol),
                label: leak_str(&plan.label),
                kind: leak_str(&plan.kind),
                count: plan.count,
            })
            .collect(),
    );
    let kernels = if program.kernels.is_empty() {
        leak_slice(synthetic_stage1_kernels(program))
    } else {
        leak_slice(
            program
                .kernels
                .iter()
                .map(|plan| KernelStage1KernelPlan {
                    symbol: leak_str(&plan.symbol),
                    relation: leak_str(&plan.relation),
                    kind: leak_str(&plan.kind),
                    backend: leak_str(&plan.backend),
                    abi: leak_str(&plan.abi),
                })
                .collect(),
        )
    };
    let claims = leak_slice(
        program
            .claims
            .iter()
            .map(|plan| KernelStage1SumcheckClaimPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                domain: leak_str(&plan.domain),
                num_rounds: plan.num_rounds,
                degree: plan.degree,
                claim: leak_str(&plan.claim),
                kernel: leak_str(stage1_kernel_symbol(
                    plan.kernel.as_deref(),
                    plan.relation.as_deref(),
                )),
                claim_value: leak_str(&plan.claim_value),
                input_openings: leak_str_slice(&plan.input_openings),
            })
            .collect(),
    );
    let batches = leak_slice(
        program
            .batches
            .iter()
            .map(|plan| KernelStage1SumcheckBatchPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                proof_slot: leak_str(&plan.proof_slot),
                policy: leak_str(&plan.policy),
                count: plan.count,
                ordered_claims: leak_str_slice(&plan.ordered_claims),
                claim_operands: leak_str_slice(&plan.claim_operands),
                claim_label: leak_str(&plan.claim_label),
                round_label: leak_str(&plan.round_label),
                round_schedule: leak_usize_slice(&plan.round_schedule),
            })
            .collect(),
    );
    let drivers = leak_slice(
        program
            .drivers
            .iter()
            .map(|plan| KernelStage1SumcheckDriverPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                proof_slot: leak_str(&plan.proof_slot),
                kernel: leak_str(stage1_kernel_symbol(
                    plan.kernel.as_deref(),
                    plan.relation.as_deref(),
                )),
                batch: leak_str(&plan.batch),
                policy: leak_str(&plan.policy),
                round_schedule: leak_usize_slice(&plan.round_schedule),
                claim_label: leak_str(&plan.claim_label),
                round_label: leak_str(&plan.round_label),
                num_rounds: plan.num_rounds,
                degree: plan.degree,
            })
            .collect(),
    );
    let evals = leak_slice(
        program
            .evals
            .iter()
            .map(|plan| KernelStage1SumcheckEvalPlan {
                symbol: leak_str(&plan.symbol),
                source: leak_str(&plan.source),
                name: leak_str(&plan.name),
                index: plan.index,
                oracle: leak_str(&plan.oracle),
            })
            .collect(),
    );
    let instance_results = leak_slice(
        program
            .instance_results
            .iter()
            .map(|plan| KernelStage1SumcheckInstanceResultPlan {
                symbol: leak_str(&plan.symbol),
                source: leak_str(&plan.source),
                claim: leak_str(&plan.claim),
                relation: leak_str(&plan.relation),
                index: plan.index,
                point_arity: plan.point_arity,
                num_rounds: plan.num_rounds,
                round_offset: plan.round_offset,
                point_order: leak_str(&plan.point_order),
                degree: plan.degree,
            })
            .collect(),
    );
    let opening_claims = leak_slice(
        program
            .opening_claims
            .iter()
            .map(|plan| KernelStage1OpeningClaimPlan {
                symbol: leak_str(&plan.symbol),
                oracle: leak_str(&plan.oracle),
                domain: leak_str(&plan.domain),
                point_arity: plan.point_arity,
                claim_kind: leak_str(&plan.claim_kind),
                point_source: leak_str(&plan.point_source),
                eval_source: leak_str(&plan.eval_source),
            })
            .collect(),
    );
    let opening_batches = leak_slice(
        program
            .opening_batches
            .iter()
            .map(|plan| KernelStage1OpeningBatchPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                proof_slot: leak_str(&plan.proof_slot),
                policy: leak_str(&plan.policy),
                count: plan.count,
                ordered_claims: leak_str_slice(&plan.ordered_claims),
                claim_operands: leak_str_slice(&plan.claim_operands),
            })
            .collect(),
    );

    Box::leak(Box::new(KernelStage1CpuProgramPlan {
        params: Stage1Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        transcript_squeezes,
        kernels,
        claims,
        batches,
        drivers,
        instance_results,
        evals,
        opening_claims,
        opening_batches,
    }))
}

fn synthetic_stage1_kernels(program: &CompilerStage1CpuProgram) -> Vec<KernelStage1KernelPlan> {
    let mut kernels: Vec<KernelStage1KernelPlan> = Vec::new();
    for driver in &program.drivers {
        let relation = driver
            .relation
            .as_deref()
            .expect("verifier driver relation");
        let kernel = synthetic_stage1_kernel(relation);
        if !kernels
            .iter()
            .any(|existing| existing.symbol == kernel.symbol)
        {
            kernels.push(kernel);
        }
    }
    kernels
}

fn stage1_kernel_symbol<'a>(kernel: Option<&'a str>, relation: Option<&str>) -> &'a str {
    if let Some(kernel) = kernel {
        return kernel;
    }
    synthetic_stage1_kernel(relation.expect("verifier relation")).symbol
}

fn synthetic_stage1_kernel(relation: &str) -> KernelStage1KernelPlan {
    match relation {
        "jolt.stage1.outer.uniskip" => KernelStage1KernelPlan {
            symbol: "jolt.cpu.stage1.outer.uniskip",
            relation: "jolt.stage1.outer.uniskip",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage1_outer_uniskip",
        },
        "jolt.stage1.outer.remaining" => KernelStage1KernelPlan {
            symbol: "jolt.cpu.stage1.outer.remaining",
            relation: "jolt.stage1.outer.remaining",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage1_outer_remaining",
        },
        relation => panic!("unsupported Stage1 verifier relation `{relation}`"),
    }
}

fn leak_stage2_product_uniskip_program(
    program: &CompilerStage2CpuProgram,
) -> &'static KernelStage2CpuProgramPlan {
    let transcript_squeeze = program
        .transcript_squeezes
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.tau_high")
        .expect("product tau squeeze");
    let challenge_extract = program
        .challenge_extracts
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.tau_high.scalar")
        .expect("product tau extract");
    let field_expr = program
        .field_exprs
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.uniskip.claim_expr")
        .expect("product claim expr");
    let kernel = program
        .kernels
        .iter()
        .find(|plan| plan.symbol == "jolt.cpu.stage2.product_virtual.uniskip")
        .expect("product kernel");
    let claim = program
        .claims
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.uniskip.input")
        .expect("product claim");
    let batch = program
        .batches
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.uniskip.batch")
        .expect("product batch");
    let driver = program
        .drivers
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.uniskip.sumcheck")
        .expect("product driver");
    let instance = program
        .instance_results
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.uniskip.instance")
        .expect("product instance");
    let eval = program
        .evals
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.uniskip.eval.UnivariateSkip")
        .expect("product eval");
    let opening_claim = program
        .opening_claims
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.uniskip.opening.UnivariateSkip")
        .expect("product opening");

    Box::leak(Box::new(KernelStage2CpuProgramPlan {
        params: Stage2Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(vec![
            KernelStage2ProgramStepPlan {
                kind: "transcript_squeeze",
                symbol: leak_str(&transcript_squeeze.symbol),
            },
            KernelStage2ProgramStepPlan {
                kind: "sumcheck_driver",
                symbol: leak_str(&driver.symbol),
            },
        ]),
        transcript_squeezes: leak_slice(vec![KernelStage2TranscriptSqueezePlan {
            symbol: leak_str(&transcript_squeeze.symbol),
            label: leak_str(&transcript_squeeze.label),
            kind: leak_str(&transcript_squeeze.kind),
            count: transcript_squeeze.count,
        }]),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .filter(|plan| {
                    matches!(
                        plan.symbol.as_str(),
                        "stage2.input.stage1.Product"
                            | "stage2.input.stage1.ShouldBranch"
                            | "stage2.input.stage1.ShouldJump"
                    )
                })
                .map(|plan| Stage2OpeningInputPlan {
                    symbol: leak_str(&plan.symbol),
                    source_stage: leak_str(&plan.source_stage),
                    source_claim: leak_str(&plan.source_claim),
                    oracle: leak_str(&plan.oracle),
                    domain: leak_str(&plan.domain),
                    point_arity: plan.point_arity,
                    claim_kind: leak_str(&plan.claim_kind),
                })
                .collect(),
        ),
        field_constants: &[] as &[KernelStage2FieldConstantPlan],
        challenge_extracts: leak_slice(vec![KernelStage2ChallengeExtractPlan {
            symbol: leak_str(&challenge_extract.symbol),
            source: leak_str(&challenge_extract.source),
            index: challenge_extract.index,
            challenge_source: leak_str(&challenge_extract.challenge_source),
        }]),
        field_exprs: leak_slice(vec![KernelStage2FieldExprPlan {
            symbol: leak_str(&field_expr.symbol),
            kind: leak_str(&field_expr.kind),
            formula: leak_str(&field_expr.formula),
            operand_names: leak_str_slice(&field_expr.operand_names),
            operands: leak_str_slice(&field_expr.operands),
        }]),
        kernels: leak_slice(vec![KernelStage2KernelPlan {
            symbol: leak_str(&kernel.symbol),
            relation: leak_str(&kernel.relation),
            kind: leak_str(&kernel.kind),
            backend: leak_str(&kernel.backend),
            abi: leak_str(&kernel.abi),
        }]),
        claims: leak_slice(vec![KernelStage2SumcheckClaimPlan {
            symbol: leak_str(&claim.symbol),
            stage: leak_str(&claim.stage),
            domain: leak_str(&claim.domain),
            num_rounds: claim.num_rounds,
            degree: claim.degree,
            claim: leak_str(&claim.claim),
            kernel: leak_str(claim.kernel.as_deref().expect("prover claim kernel")),
            claim_value: leak_str(&claim.claim_value),
            input_openings: leak_str_slice(&claim.input_openings),
        }]),
        batches: leak_slice(vec![KernelStage2SumcheckBatchPlan {
            symbol: leak_str(&batch.symbol),
            stage: leak_str(&batch.stage),
            proof_slot: leak_str(&batch.proof_slot),
            policy: leak_str(&batch.policy),
            count: batch.count,
            ordered_claims: leak_str_slice(&batch.ordered_claims),
            claim_operands: leak_str_slice(&batch.claim_operands),
            claim_label: leak_str(&batch.claim_label),
            round_label: leak_str(&batch.round_label),
            round_schedule: leak_usize_slice(&batch.round_schedule),
        }]),
        drivers: leak_slice(vec![KernelStage2SumcheckDriverPlan {
            symbol: leak_str(&driver.symbol),
            stage: leak_str(&driver.stage),
            proof_slot: leak_str(&driver.proof_slot),
            kernel: leak_str(driver.kernel.as_deref().expect("prover driver kernel")),
            batch: leak_str(&driver.batch),
            policy: leak_str(&driver.policy),
            round_schedule: leak_usize_slice(&driver.round_schedule),
            claim_label: leak_str(&driver.claim_label),
            round_label: leak_str(&driver.round_label),
            num_rounds: driver.num_rounds,
            degree: driver.degree,
        }]),
        instance_results: leak_slice(vec![Stage2SumcheckInstanceResultPlan {
            symbol: leak_str(&instance.symbol),
            source: leak_str(&instance.source),
            claim: leak_str(&instance.claim),
            relation: leak_str(&instance.relation),
            index: instance.index,
            point_arity: instance.point_arity,
            num_rounds: instance.num_rounds,
            round_offset: instance.round_offset,
            point_order: leak_str(&instance.point_order),
            degree: instance.degree,
        }]),
        evals: leak_slice(vec![KernelStage2SumcheckEvalPlan {
            symbol: leak_str(&eval.symbol),
            source: leak_str(&eval.source),
            name: leak_str(&eval.name),
            index: eval.index,
            oracle: leak_str(&eval.oracle),
        }]),
        point_slices: &[] as &[Stage2PointSlicePlan],
        point_concats: &[] as &[Stage2PointConcatPlan],
        opening_claims: leak_slice(vec![Stage2OpeningClaimPlan {
            symbol: leak_str(&opening_claim.symbol),
            oracle: leak_str(&opening_claim.oracle),
            domain: leak_str(&opening_claim.domain),
            point_arity: opening_claim.point_arity,
            claim_kind: leak_str(&opening_claim.claim_kind),
            point_source: leak_str(&opening_claim.point_source),
            eval_source: leak_str(&opening_claim.eval_source),
        }]),
        opening_batches: &[] as &[Stage2OpeningBatchPlan],
    }))
}

fn leak_stage2_program(program: &CompilerStage2CpuProgram) -> &'static KernelStage2CpuProgramPlan {
    Box::leak(Box::new(KernelStage2CpuProgramPlan {
        params: Stage2Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| KernelStage2ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| KernelStage2TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| Stage2OpeningInputPlan {
                    symbol: leak_str(&plan.symbol),
                    source_stage: leak_str(&plan.source_stage),
                    source_claim: leak_str(&plan.source_claim),
                    oracle: leak_str(&plan.oracle),
                    domain: leak_str(&plan.domain),
                    point_arity: plan.point_arity,
                    claim_kind: leak_str(&plan.claim_kind),
                })
                .collect(),
        ),
        field_constants: leak_slice(
            program
                .field_constants
                .iter()
                .map(|plan| KernelStage2FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        challenge_extracts: leak_slice(
            program
                .challenge_extracts
                .iter()
                .map(|plan| KernelStage2ChallengeExtractPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    index: plan.index,
                    challenge_source: leak_str(&plan.challenge_source),
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| KernelStage2FieldExprPlan {
                    symbol: leak_str(&plan.symbol),
                    kind: leak_str(&plan.kind),
                    formula: leak_str(&plan.formula),
                    operand_names: leak_str_slice(&plan.operand_names),
                    operands: leak_str_slice(&plan.operands),
                })
                .collect(),
        ),
        kernels: leak_slice(
            program
                .kernels
                .iter()
                .map(|plan| KernelStage2KernelPlan {
                    symbol: leak_str(&plan.symbol),
                    relation: leak_str(&plan.relation),
                    kind: leak_str(&plan.kind),
                    backend: leak_str(&plan.backend),
                    abi: leak_str(&plan.abi),
                })
                .collect(),
        ),
        claims: leak_slice(
            program
                .claims
                .iter()
                .map(|plan| KernelStage2SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage2 claim kernel")),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| KernelStage2SumcheckBatchPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    policy: leak_str(&plan.policy),
                    count: plan.count,
                    ordered_claims: leak_str_slice(&plan.ordered_claims),
                    claim_operands: leak_str_slice(&plan.claim_operands),
                    claim_label: leak_str(&plan.claim_label),
                    round_label: leak_str(&plan.round_label),
                    round_schedule: leak_usize_slice(&plan.round_schedule),
                })
                .collect(),
        ),
        drivers: leak_slice(
            program
                .drivers
                .iter()
                .map(|plan| KernelStage2SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage2 driver kernel")),
                    batch: leak_str(&plan.batch),
                    policy: leak_str(&plan.policy),
                    round_schedule: leak_usize_slice(&plan.round_schedule),
                    claim_label: leak_str(&plan.claim_label),
                    round_label: leak_str(&plan.round_label),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                })
                .collect(),
        ),
        instance_results: leak_slice(
            program
                .instance_results
                .iter()
                .map(|plan| Stage2SumcheckInstanceResultPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    claim: leak_str(&plan.claim),
                    relation: leak_str(&plan.relation),
                    index: plan.index,
                    point_arity: plan.point_arity,
                    num_rounds: plan.num_rounds,
                    round_offset: plan.round_offset,
                    point_order: leak_str(&plan.point_order),
                    degree: plan.degree,
                })
                .collect(),
        ),
        evals: leak_slice(
            program
                .evals
                .iter()
                .map(|plan| KernelStage2SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| Stage2PointSlicePlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    offset: plan.offset,
                    length: plan.length,
                    input: leak_str(&plan.input),
                })
                .collect(),
        ),
        point_concats: leak_slice(
            program
                .point_concats
                .iter()
                .map(|plan| Stage2PointConcatPlan {
                    symbol: leak_str(&plan.symbol),
                    layout: leak_str(&plan.layout),
                    arity: plan.arity,
                    inputs: leak_str_slice(&plan.inputs),
                })
                .collect(),
        ),
        opening_claims: leak_slice(
            program
                .opening_claims
                .iter()
                .map(|plan| Stage2OpeningClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    oracle: leak_str(&plan.oracle),
                    domain: leak_str(&plan.domain),
                    point_arity: plan.point_arity,
                    claim_kind: leak_str(&plan.claim_kind),
                    point_source: leak_str(&plan.point_source),
                    eval_source: leak_str(&plan.eval_source),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| Stage2OpeningBatchPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    policy: leak_str(&plan.policy),
                    count: plan.count,
                    ordered_claims: leak_str_slice(&plan.ordered_claims),
                    claim_operands: leak_str_slice(&plan.claim_operands),
                })
                .collect(),
        ),
    }))
}

fn leak_str(value: &str) -> &'static str {
    Box::leak(value.to_owned().into_boxed_str())
}

fn leak_str_slice(values: &[String]) -> &'static [&'static str] {
    let leaked = values
        .iter()
        .map(|value| leak_str(value))
        .collect::<Vec<_>>();
    Box::leak(leaked.into_boxed_slice())
}

fn leak_usize_slice(values: &[usize]) -> &'static [usize] {
    Box::leak(values.to_vec().into_boxed_slice())
}

fn leak_slice<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}

fn core_muldiv_commitment_fixture() -> CoreMuldivCommitmentFixture {
    DoryGlobals::reset();

    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("muldiv inputs");
    let mut core_program = host::Program::new("muldiv-guest");
    let (core_bytecode, init_memory_state, _, entry_address) = core_program.decode();
    let (_, _, _, core_io_device) = core_program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        core_io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        entry_address,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = core_program.get_elf_contents().expect("muldiv elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io = prover.program_io.clone();
    let initial_ram_state = prover.initial_ram_state.clone();
    let final_ram_state = prover.final_ram_state.clone();
    let (proof, _debug): (CoreProof, _) = prover.prove();
    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));

    let mut host_program = Program::new("muldiv-guest");
    let (bytecode_raw, _, _, host_entry_address) = host_program.decode();
    let (_, trace, _, host_io_device) = host_program.trace(&inputs, &[], &[]);
    let mut padded_trace = trace.clone();
    padded_trace.resize(proof.trace_length, jolt_host::Cycle::NoOp);
    let product_virtual_cycles = (0..proof.trace_length)
        .map(|index| {
            let row = CoreProductCycleInputs::from_trace::<CoreFr>(&padded_trace, index);
            Stage2ProductVirtualCycle {
                instruction_left_input: row.instruction_left_input,
                instruction_right_input: row.instruction_right_input,
                should_branch_lookup_output: row.should_branch_lookup_output,
                write_lookup_output_to_rd_flag: row.write_lookup_output_to_rd_flag,
                jump_flag: row.jump_flag,
                should_branch_flag: row.should_branch_flag,
                not_next_noop: row.not_next_noop,
                virtual_instruction_flag: row.virtual_instruction_flag,
            }
        })
        .collect();
    let instruction_lookup_cycles = padded_trace
        .iter()
        .map(|cycle| {
            let (left_instruction_input, right_instruction_input) =
                LookupQuery::<XLEN>::to_instruction_inputs(cycle);
            let (left_lookup_operand, right_lookup_operand) =
                LookupQuery::<XLEN>::to_lookup_operands(cycle);
            let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);
            Stage2InstructionLookupCycle {
                lookup_output,
                left_lookup_operand,
                right_lookup_operand,
                left_instruction_input,
                right_instruction_input,
            }
        })
        .collect();
    let ram_accesses = padded_trace
        .iter()
        .map(|cycle| match cycle.ram_access() {
            RAMAccess::Read(read) => Stage2RamAccess {
                remapped_address: remap_address(read.address, &host_io_device.memory_layout)
                    .map(|address| address as usize),
                read_value: read.value,
                write_value: read.value,
            },
            RAMAccess::Write(write) => Stage2RamAccess {
                remapped_address: remap_address(write.address, &host_io_device.memory_layout)
                    .map(|address| address as usize),
                read_value: write.pre_value,
                write_value: write.post_value,
            },
            RAMAccess::NoOp => Stage2RamAccess::noop(),
        })
        .collect();
    let ram_start_address = host_io_device.memory_layout.get_lowest_address();
    let ram_output_layout = Stage2RamOutputLayout {
        io_start: remap_address(
            host_io_device.memory_layout.input_start,
            &host_io_device.memory_layout,
        )
        .expect("input start remaps") as usize,
        io_end: remap_address(RAM_START_ADDRESS, &host_io_device.memory_layout)
            .expect("RAM start remaps") as usize,
    };
    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, host_entry_address);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), proof.trace_length);
    let (cycle_inputs, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        proof.trace_length,
        &bytecode,
        &host_io_device.memory_layout,
        r1cs_key.num_vars_padded,
    );

    let log_t = proof.trace_length.trailing_zeros() as usize;
    let log_k_bytecode = prover_preprocessing
        .shared
        .bytecode
        .code_size
        .trailing_zeros() as usize;
    let log_k_ram = proof.ram_K.trailing_zeros() as usize;
    let params = JoltProtocolParams::new(log_t, log_k_bytecode, log_k_ram);

    let commitments = proof.commitments.clone();

    CoreMuldivCommitmentFixture {
        params,
        pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
        proof,
        verifier_preprocessing,
        io,
        entry_address,
        cycle_inputs,
        r1cs_witness,
        product_virtual_cycles,
        instruction_lookup_cycles,
        ram_accesses,
        initial_ram_state,
        final_ram_state,
        ram_start_address,
        ram_output_layout,
        commitments,
    }
}

fn run_bolt_commitment_prover(program: &CommitmentCpuProgram) -> BoltCommitmentTrace {
    assert_eq!(program.role, Role::Prover);
    let setup = DoryScheme::setup_prover(max_num_vars(program));
    run_bolt_commitment_prover_with(program, &setup, |oracle, num_vars| {
        optional_oracle_data(oracle, num_vars)
    })
}

fn run_bolt_commitment_prover_with<F>(
    program: &CommitmentCpuProgram,
    setup: &DoryProverSetup,
    mut materialize: F,
) -> BoltCommitmentTrace
where
    F: FnMut(&str, usize) -> Option<Vec<Fr>>,
{
    assert_eq!(program.role, Role::Prover);
    let mut commitments = Vec::new();
    let mut records = Vec::new();

    for plan in &program.batch_plans {
        assert_eq!(plan.count, plan.oracles.len());
        for oracle in &plan.oracles {
            let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
            let data = materialize(oracle, oracle_num_vars)
                .unwrap_or_else(|| panic!("missing batch oracle `{oracle}`"));
            let data = into_padded_oracle(data, oracle_num_vars);
            let (commitment, _) = commit_with_layout(&data, plan.num_vars, setup);
            records.push(CommitmentRecord {
                artifact: plan.artifact.clone(),
            });
            commitments.push(Some(commitment));
        }
    }
    for plan in &program.optional_plans {
        let oracle_num_vars = oracle_num_vars(program, &plan.oracle, plan.num_vars);
        let commitment = materialize(&plan.oracle, oracle_num_vars)
            .filter(|data| !should_skip_optional(&plan.skip_policy, data))
            .map(|data| {
                let data = into_padded_oracle(data, oracle_num_vars);
                commit_with_layout(&data, plan.num_vars, setup).0
            });
        records.push(CommitmentRecord {
            artifact: plan.artifact.clone(),
        });
        commitments.push(commitment);
    }

    let log = bolt_transcript_log(&records, &commitments, &program.transcript_steps);
    BoltCommitmentTrace {
        commitments,
        records,
        log,
    }
}

fn real_muldiv_oracle_data(
    program: &CommitmentCpuProgram,
    cycle_inputs: &[CycleInput],
) -> BTreeMap<String, Option<Vec<Fr>>> {
    let mut data = BTreeMap::new();
    for plan in &program.oracle_plans {
        let materialized = match &plan.generation {
            OracleGeneration::Reference => continue,
            OracleGeneration::DenseTrace { .. } => {
                let values = dense_source(cycle_inputs, &plan.source);
                Some(dense_i128_column_to_field(
                    &values,
                    target_len(plan.num_vars),
                ))
            }
            OracleGeneration::OneHotChunk {
                trace_num_vars,
                chunk,
                num_chunks,
                chunk_bits,
                padding,
                ..
            } => {
                let values = one_hot_source(cycle_inputs, &plan.source);
                Some(one_hot_chunk_address_major(
                    &values,
                    *chunk,
                    *num_chunks,
                    *chunk_bits,
                    target_len(*trace_num_vars),
                    padding_value(padding),
                ))
            }
            OracleGeneration::OptionalAdvice { .. } => {
                optional_field_oracle::<Fr>(None, target_len(plan.num_vars))
            }
        };
        let _ = data.insert(plan.oracle.clone(), materialized);
    }
    data
}

fn run_bolt_commitment_verifier(
    program: &CommitmentCpuProgram,
    proof_commitments: &[Option<DoryCommitment>],
) -> BoltCommitmentTrace {
    assert_eq!(program.role, Role::Verifier);
    let mut commitments = Vec::new();
    let mut records = Vec::new();
    let mut cursor = 0;

    for plan in &program.batch_plans {
        assert_eq!(plan.count, plan.oracles.len());
        for _ in &plan.oracles {
            let commitment = proof_commitments
                .get(cursor)
                .expect("proof commitment slot")
                .clone();
            assert!(commitment.is_some(), "batch commitments cannot be skipped");
            cursor += 1;
            records.push(CommitmentRecord {
                artifact: plan.artifact.clone(),
            });
            commitments.push(commitment);
        }
    }
    for plan in &program.optional_plans {
        let commitment = proof_commitments
            .get(cursor)
            .expect("optional proof commitment slot")
            .clone();
        cursor += 1;
        records.push(CommitmentRecord {
            artifact: plan.artifact.clone(),
        });
        commitments.push(commitment);
    }
    assert_eq!(cursor, proof_commitments.len());

    let log = bolt_transcript_log(&records, &commitments, &program.transcript_steps);
    BoltCommitmentTrace {
        commitments,
        records,
        log,
    }
}

fn bolt_transcript_log(
    records: &[CommitmentRecord],
    commitments: &[Option<DoryCommitment>],
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let mut transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);

    for step in transcript_steps {
        let mut appended = false;
        for (record, commitment) in records.iter().zip(commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(
                    static_transcript_label(&step.label),
                    commitment.serialized_len(),
                ));
                commitment.append_to_transcript(&mut transcript);
                appended = true;
            }
        }
        assert!(step.optional || appended, "missing transcript source");
    }

    transcript.into_log()
}

fn append_bolt_preamble<T>(
    transcript: &mut T,
    program_io: &JoltDevice,
    ram_k: usize,
    trace_length: usize,
    entry_address: u64,
) where
    T: Transcript<Challenge = Fr>,
{
    append_u64(
        transcript,
        b"max_input_size",
        program_io.memory_layout.max_input_size,
    );
    append_u64(
        transcript,
        b"max_output_size",
        program_io.memory_layout.max_output_size,
    );
    append_u64(transcript, b"heap_size", program_io.memory_layout.heap_size);
    append_bytes(transcript, b"inputs", &program_io.inputs);
    append_bytes(transcript, b"outputs", &program_io.outputs);
    append_u64(transcript, b"panic", program_io.panic as u64);
    append_u64(transcript, b"ram_K", ram_k as u64);
    append_u64(transcript, b"trace_length", trace_length as u64);
    append_u64(transcript, b"entry_address", entry_address);
}

fn append_u64<T>(transcript: &mut T, label: &'static [u8], value: u64)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label));
    transcript.append(&U64Word(value));
}

fn append_bytes<T>(transcript: &mut T, label: &'static [u8], bytes: &[u8])
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(label, bytes.len() as u64));
    transcript.append_bytes(bytes);
}

fn append_bolt_commitments_to_transcript<T>(
    transcript: &mut T,
    records: &[CommitmentRecord],
    commitments: &[Option<DoryCommitment>],
    transcript_steps: &[TranscriptStep],
) where
    T: Transcript<Challenge = Fr>,
{
    for step in transcript_steps {
        let mut appended = false;
        for (record, commitment) in records.iter().zip(commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(
                    static_transcript_label(&step.label),
                    commitment.serialized_len(),
                ));
                commitment.append_to_transcript(transcript);
                appended = true;
            }
        }
        assert!(step.optional || appended, "missing transcript source");
    }
}

fn assert_core_accepts_bolt_stage1(
    fixture: &CoreMuldivCommitmentFixture,
    artifacts: &Stage1ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&artifacts.sumchecks[1].proof.round_polynomials);

    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        proof,
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier
        .verify_stage1()
        .expect("jolt-core accepts Bolt Stage 1 proof");
}

fn assert_core_accepts_bolt_stage2(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    assert_core_stage2_opening_claims_match_bolt(&fixture.proof, stage2_artifacts);
    assert_core_stage2_sumcheck_proof_matches_bolt(&fixture.proof, &stage2_artifacts.sumchecks[1]);

    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        proof,
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core accepts Bolt Stage 1");
    let _ = verifier.verify_stage2().expect("core accepts Bolt Stage 2");
}

fn assert_core_stage2_opening_claims_match_bolt(
    proof: &CoreProof,
    artifacts: &Stage2ExecutionArtifacts<Fr>,
) {
    let expected = [
        (
            "stage2.product_virtual.uniskip.eval.UnivariateSkip",
            OpeningId::virt(
                VirtualPolynomial::UnivariateSkip,
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.ram_read_write.eval.RamVal",
            OpeningId::virt(VirtualPolynomial::RamVal, SumcheckId::RamReadWriteChecking),
        ),
        (
            "stage2.ram_read_write.eval.RamRa",
            OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamReadWriteChecking),
        ),
        (
            "stage2.ram_read_write.eval.RamInc",
            OpeningId::committed(
                CommittedPolynomial::RamInc,
                SumcheckId::RamReadWriteChecking,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.LeftInstructionInput",
            OpeningId::virt(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.RightInstructionInput",
            OpeningId::virt(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.OpFlagJump",
            OpeningId::virt(
                VirtualPolynomial::OpFlags(CircuitFlags::Jump),
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD",
            OpeningId::virt(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.LookupOutput",
            OpeningId::virt(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.InstructionFlagBranch",
            OpeningId::virt(
                VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.NextIsNoop",
            OpeningId::virt(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction",
            OpeningId::virt(
                VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
                SumcheckId::SpartanProductVirtualization,
            ),
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
            OpeningId::virt(
                VirtualPolynomial::LookupOutput,
                SumcheckId::InstructionClaimReduction,
            ),
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
            OpeningId::virt(
                VirtualPolynomial::LeftLookupOperand,
                SumcheckId::InstructionClaimReduction,
            ),
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            OpeningId::virt(
                VirtualPolynomial::RightLookupOperand,
                SumcheckId::InstructionClaimReduction,
            ),
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            OpeningId::virt(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::InstructionClaimReduction,
            ),
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            OpeningId::virt(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::InstructionClaimReduction,
            ),
        ),
        (
            "stage2.ram_raf.eval.RamRa",
            OpeningId::virt(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation),
        ),
        (
            "stage2.ram_output.eval.RamValFinal",
            OpeningId::virt(VirtualPolynomial::RamValFinal, SumcheckId::RamOutputCheck),
        ),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();

    for output in &artifacts.sumchecks {
        for eval in &output.evals {
            let Some(opening_id) = expected.get(eval.name) else {
                continue;
            };
            if let Some((_, core_claim)) = proof.opening_claims.0.get(opening_id) {
                assert_eq!(
                    eval.value,
                    Fr::from(*core_claim),
                    "Stage 2 opening claim mismatch for {}",
                    eval.name
                );
            }
        }
    }
}

struct BoltStage2ChainVerifierInput<'a> {
    fixture: &'a CoreMuldivCommitmentFixture,
    commitment_verifier_program: &'a CommitmentCpuProgram,
    commitment_verifier_trace: &'a BoltCommitmentTrace,
    stage1_prover_plan: &'static KernelStage1CpuProgramPlan,
    stage2_prover_plan: &'static KernelStage2CpuProgramPlan,
    stage1_artifacts: &'a Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &'a Stage2ExecutionArtifacts<Fr>,
    prover_transcript: &'a CheckpointTranscript<jolt_transcript::Blake2bTranscript<Fr>>,
}

fn assert_bolt_chain_verifier_accepts_stage2_product_uniskip(
    input: BoltStage2ChainVerifierInput<'_>,
) {
    let mut verifier_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(
        &mut verifier_transcript,
        &input.fixture.io,
        input.fixture.proof.ram_K,
        input.fixture.proof.trace_length,
        input.fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        &mut verifier_transcript,
        &input.commitment_verifier_trace.records,
        &input.commitment_verifier_trace.commitments,
        &input.commitment_verifier_program.transcript_steps,
    );

    let stage1_proof = Stage1Proof::from(input.stage1_artifacts.clone());
    let mut stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
    let verified_stage1 = execute_stage1_program(
        input.stage1_prover_plan,
        Stage1ExecutionMode::Verifier,
        &mut stage1_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts Bolt prover proof");
    assert_eq!(
        input.stage1_artifacts.sumchecks.len(),
        verified_stage1.sumchecks.len()
    );
    assert_eq!(
        input.stage1_artifacts.opening_values.len(),
        verified_stage1.opening_values.len()
    );

    let stage2_openings = stage2_product_opening_inputs(&verified_stage1);
    let stage2_proof = Stage2Proof::from(input.stage2_artifacts.clone());
    let mut stage2_verifier = Stage2VerifierKernelExecutor::new(&stage2_proof, &stage2_openings);
    let verified_stage2 = execute_stage2_program(
        input.stage2_prover_plan,
        Stage2ExecutionMode::Verifier,
        &mut stage2_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts Bolt prover proof");

    assert_eq!(
        input.stage2_artifacts.sumchecks.len(),
        verified_stage2.sumchecks.len()
    );
    assert_eq!(
        input.stage2_artifacts.sumchecks[0].point,
        verified_stage2.sumchecks[0].point
    );
    assert_eq!(
        input.stage2_artifacts.sumchecks[0].evals[0].value,
        verified_stage2.sumchecks[0].evals[0].value
    );
    assert_state_history_match(
        &transcript_states(input.prover_transcript.log()),
        &transcript_states(verifier_transcript.log()),
    );
}

fn stage2_product_opening_inputs(
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
) -> Vec<Stage2OpeningInputValue<Fr>> {
    ["Product", "ShouldBranch", "ShouldJump"]
        .into_iter()
        .map(|oracle| {
            let source_claim = match oracle {
                "Product" => "stage1.outer_remaining.opening.Product",
                "ShouldBranch" => "stage1.outer_remaining.opening.ShouldBranch",
                "ShouldJump" => "stage1.outer_remaining.opening.ShouldJump",
                _ => unreachable!(),
            };
            let opening = stage1_artifacts
                .opening_value(source_claim)
                .unwrap_or_else(|| panic!("missing Stage 1 opening {source_claim}"));
            Stage2OpeningInputValue {
                symbol: match oracle {
                    "Product" => "stage2.input.stage1.Product",
                    "ShouldBranch" => "stage2.input.stage1.ShouldBranch",
                    "ShouldJump" => "stage2.input.stage1.ShouldJump",
                    _ => unreachable!(),
                },
                point: opening.point.clone(),
                eval: opening.eval,
            }
        })
        .collect()
}

fn stage2_opening_inputs(
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
) -> Vec<Stage2OpeningInputValue<Fr>> {
    [
        "Product",
        "ShouldBranch",
        "ShouldJump",
        "RamReadValue",
        "RamWriteValue",
        "LookupOutput",
        "LeftLookupOperand",
        "RightLookupOperand",
        "LeftInstructionInput",
        "RightInstructionInput",
        "RamAddress",
    ]
    .into_iter()
    .map(|oracle| {
        let source_claim = format!("stage1.outer_remaining.opening.{oracle}");
        let opening = stage1_artifacts
            .opening_value(&source_claim)
            .unwrap_or_else(|| panic!("missing Stage 1 opening {source_claim}"));
        Stage2OpeningInputValue {
            symbol: leak_str(&format!("stage2.input.stage1.{oracle}")),
            point: opening.point.clone(),
            eval: opening.eval,
        }
    })
    .collect()
}

fn assert_core_stage2_uniskip_proof_matches_bolt(
    proof: &CoreProof,
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) {
    let core_coefficients = match &proof.stage2_uni_skip_first_round_proof {
        UniSkipFirstRoundProofVariant::Standard(proof) => proof
            .uni_poly
            .coeffs
            .iter()
            .copied()
            .map(Fr::from)
            .collect::<Vec<_>>(),
        UniSkipFirstRoundProofVariant::Zk(_) => panic!("standard Stage 2 proof expected"),
    };
    assert_eq!(output.proof.round_polynomials.len(), 1);
    assert_eq!(
        output.proof.round_polynomials[0].coefficients(),
        core_coefficients.as_slice()
    );
}

fn assert_core_stage2_sumcheck_proof_matches_bolt(
    proof: &CoreProof,
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) {
    let core_polys = match &proof.stage2_sumcheck_proof {
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(proof) => {
            &proof.compressed_polys
        }
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Zk(_) => {
            panic!("standard Stage 2 proof expected")
        }
    };
    assert_eq!(
        core_polys.len(),
        output.proof.round_polynomials.len(),
        "Stage 2 round count mismatch"
    );
    for (round, (core, bolt)) in core_polys
        .iter()
        .zip(&output.proof.round_polynomials)
        .enumerate()
    {
        let bolt_coeffs = bolt.compress();
        let bolt_coeffs = bolt_coeffs
            .coeffs_except_linear_term()
            .iter()
            .copied()
            .map(to_ark)
            .collect::<Vec<_>>();
        assert_eq!(
            core.coeffs_except_linear_term, bolt_coeffs,
            "Stage 2 compressed coefficient mismatch at round {round}"
        );
    }
}

fn assert_core_states_match_bolt_stage1(
    fixture: &CoreMuldivCommitmentFixture,
    bolt_log: &[TranscriptEvent],
) {
    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        clone_core_proof(&fixture.proof),
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core Stage 1 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_state_history_match(&core_states, &bolt_states);
}

fn assert_core_states_match_bolt_stage2(
    fixture: &CoreMuldivCommitmentFixture,
    bolt_log: &[TranscriptEvent],
) {
    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        clone_core_proof(&fixture.proof),
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core Stage 1 verifies");
    let _ = verifier.verify_stage2().expect("core Stage 2 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_state_history_match(&core_states, &bolt_states);
}

fn assert_bolt_stage1_tamper_rejected(
    stage1_verifier_plan: &'static KernelStage1CpuProgramPlan,
    proof: &Stage1Proof<Fr>,
    fixture: &CoreMuldivCommitmentFixture,
    commitment_verifier_trace: &BoltCommitmentTrace,
    transcript_steps: &[TranscriptStep],
) {
    let mut tampered = proof.clone();
    let round_poly = &mut tampered.sumchecks[1].proof.round_polynomials[0];
    let mut coefficients = round_poly.coefficients().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round_poly = UnivariatePoly::new(coefficients);

    let mut transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(
        &mut transcript,
        &fixture.io,
        fixture.proof.ram_K,
        fixture.proof.trace_length,
        fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        &mut transcript,
        &commitment_verifier_trace.records,
        &commitment_verifier_trace.commitments,
        transcript_steps,
    );

    let mut verifier = Stage1VerifierKernelExecutor::new(&tampered);
    let error = execute_stage1_program(
        stage1_verifier_plan,
        Stage1ExecutionMode::Verifier,
        &mut verifier,
        &mut transcript,
    )
    .expect_err("tampered Bolt Stage 1 proof must be rejected");
    assert!(
        matches!(error, Stage1KernelError::InvalidProof { .. }),
        "unexpected tamper rejection: {error}"
    );
}

fn to_core_uniskip_proof(
    output: &jolt_kernels::stage1::Stage1SumcheckOutput<Fr>,
) -> UniSkipFirstRoundProofVariant<CoreFr, Bn254Curve, CoreBlake2bTranscript> {
    assert_eq!(output.proof.round_polynomials.len(), 1);
    let coefficients = output.proof.round_polynomials[0]
        .coefficients()
        .iter()
        .copied()
        .map(to_ark)
        .collect();
    UniSkipFirstRoundProofVariant::Standard(UniSkipFirstRoundProof::new(UniPoly::from_coeff(
        coefficients,
    )))
}

fn to_core_stage2_uniskip_proof(
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) -> UniSkipFirstRoundProofVariant<CoreFr, Bn254Curve, CoreBlake2bTranscript> {
    assert_eq!(output.proof.round_polynomials.len(), 1);
    let coefficients = output.proof.round_polynomials[0]
        .coefficients()
        .iter()
        .copied()
        .map(to_ark)
        .collect();
    UniSkipFirstRoundProofVariant::Standard(UniSkipFirstRoundProof::new(UniPoly::from_coeff(
        coefficients,
    )))
}

fn clone_core_proof(proof: &CoreProof) -> CoreProof {
    CoreJoltProof {
        commitments: proof.commitments.clone(),
        stage1_uni_skip_first_round_proof: proof.stage1_uni_skip_first_round_proof.clone(),
        stage1_sumcheck_proof: proof.stage1_sumcheck_proof.clone(),
        stage2_uni_skip_first_round_proof: proof.stage2_uni_skip_first_round_proof.clone(),
        stage2_sumcheck_proof: proof.stage2_sumcheck_proof.clone(),
        stage3_sumcheck_proof: proof.stage3_sumcheck_proof.clone(),
        stage4_sumcheck_proof: proof.stage4_sumcheck_proof.clone(),
        stage5_sumcheck_proof: proof.stage5_sumcheck_proof.clone(),
        stage6_sumcheck_proof: proof.stage6_sumcheck_proof.clone(),
        stage7_sumcheck_proof: proof.stage7_sumcheck_proof.clone(),
        joint_opening_proof: proof.joint_opening_proof.clone(),
        untrusted_advice_commitment: proof.untrusted_advice_commitment,
        opening_claims: CoreClaims(proof.opening_claims.0.clone()),
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        rw_config: proof.rw_config.clone(),
        one_hot_config: proof.one_hot_config.clone(),
        dory_layout: proof.dory_layout,
    }
}

fn transcript_states(log: &[TranscriptEvent]) -> Vec<[u8; 32]> {
    log.iter()
        .map(|event| match event {
            TranscriptEvent::Append { state_after, .. }
            | TranscriptEvent::Squeeze { state_after } => *state_after,
        })
        .collect()
}

fn assert_state_history_match(expected: &[[u8; 32]], actual: &[[u8; 32]]) {
    let min_len = expected.len().min(actual.len());
    for index in 0..min_len {
        assert_eq!(
            expected[index], actual[index],
            "transcript state mismatch at op #{index}"
        );
    }
    assert_eq!(
        expected.len(),
        actual.len(),
        "transcript state count mismatch"
    );
}

fn core_commitment_log(
    bolt_trace: &BoltCommitmentTrace,
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let mut transcript = CoreBlake2bTranscript::new(TRANSCRIPT_LABEL);
    let mut events = Vec::new();

    for step in transcript_steps {
        let mut appended = false;
        for (record, commitment) in bolt_trace.records.iter().zip(&bolt_trace.commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                let core_commitment = commitment_to_ark(commitment);
                let label = static_transcript_label(&step.label);
                let bytes = core_append_serializable_bytes(label, &core_commitment);
                let state_count_before = transcript.state_history.len();
                transcript.append_serializable(label, &core_commitment);
                let new_states = &transcript.state_history[state_count_before..];
                assert_eq!(new_states.len(), bytes.len());
                for (bytes, state_after) in bytes.into_iter().zip(new_states) {
                    events.push(TranscriptEvent::Append {
                        bytes,
                        state_after: *state_after,
                    });
                }
                appended = true;
            }
        }
        assert!(step.optional || appended, "missing core transcript source");
    }

    events
}

fn core_commitments_transcript_log(
    commitments: &[CoreCommitment],
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let mut transcript = CoreBlake2bTranscript::new(TRANSCRIPT_LABEL);
    let mut events = Vec::new();

    for step in transcript_steps {
        if step.source != "jolt.main_witness_commitments" {
            assert!(step.optional, "unexpected non-main commitment source");
            continue;
        }
        for commitment in commitments {
            let label = static_transcript_label(&step.label);
            let bytes = core_append_serializable_bytes(label, commitment);
            let state_count_before = transcript.state_history.len();
            transcript.append_serializable(label, commitment);
            let new_states = &transcript.state_history[state_count_before..];
            assert_eq!(new_states.len(), bytes.len());
            for (bytes, state_after) in bytes.into_iter().zip(new_states) {
                events.push(TranscriptEvent::Append {
                    bytes,
                    state_after: *state_after,
                });
            }
        }
    }

    events
}

fn core_append_serializable_bytes<T: CanonicalSerialize>(
    label: &'static [u8],
    data: &T,
) -> Vec<Vec<u8>> {
    let mut payload = Vec::new();
    data.serialize_uncompressed(&mut payload)
        .expect("core commitment serialization");
    let mut header = [0u8; 32];
    header[..label.len()].copy_from_slice(label);
    header[24..].copy_from_slice(&(payload.len() as u64).to_be_bytes());
    payload.reverse();
    vec![header.to_vec(), payload]
}

fn deterministic_oracle_data(oracle: &str, num_vars: usize) -> Vec<Fr> {
    let seed = oracle.bytes().fold(17u64, |state, byte| {
        state.wrapping_mul(131).wrapping_add(byte as u64)
    });
    (0..target_len(num_vars))
        .map(|index| Fr::from_u64(seed.wrapping_add(index as u64 + 1)))
        .collect()
}

fn dense_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<i128> {
    let slot = match source {
        "trace.rd_inc" => 0,
        "trace.ram_inc" => 1,
        _ => panic!("unsupported dense source `{source}`"),
    };
    cycle_inputs.iter().map(|cycle| cycle.dense[slot]).collect()
}

fn one_hot_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<Option<u128>> {
    let slot = match source {
        "trace.instruction_keys" => 0,
        "trace.bytecode_indices" => 1,
        "trace.ram_addresses" => 2,
        _ => panic!("unsupported one-hot source `{source}`"),
    };
    cycle_inputs
        .iter()
        .map(|cycle| cycle.one_hot[slot])
        .collect()
}

fn padding_value(padding: &str) -> Option<u128> {
    match padding {
        "zero" => Some(0),
        "none" => None,
        _ => panic!("unsupported padding `{padding}`"),
    }
}

fn optional_oracle_data(oracle: &str, num_vars: usize) -> Option<Vec<Fr>> {
    match oracle {
        "UntrustedAdvice" | "TrustedAdvice" => None,
        _ => Some(deterministic_oracle_data(oracle, num_vars)),
    }
}

fn should_skip_optional(policy: &OptionalSkipPolicy, data: &[Fr]) -> bool {
    match policy {
        OptionalSkipPolicy::MissingOrZero => data.iter().all(|value| *value == Fr::from_u64(0)),
    }
}

fn oracle_num_vars(program: &CommitmentCpuProgram, oracle: &str, fallback: usize) -> usize {
    program
        .oracle_plans
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn into_padded_oracle(mut data: Vec<Fr>, num_vars: usize) -> Vec<Fr> {
    let target_len = target_len(num_vars);
    assert!(
        data.len() <= target_len,
        "oracle has {} values, target length is {target_len}",
        data.len()
    );
    data.resize(target_len, Fr::from_u64(0));
    data
}

fn commit_with_layout(
    data: &[Fr],
    layout_num_vars: usize,
    setup: &DoryProverSetup,
) -> (DoryCommitment, DoryHint) {
    let row_len = target_len(layout_num_vars.div_ceil(2));
    let mut partial = DoryScheme::begin(setup);
    for row in data.chunks(row_len) {
        DoryScheme::feed(&mut partial, row, setup);
    }
    let hint = DoryHint(partial.row_commitments.clone());
    let commitment = DoryScheme::finish(partial, setup);
    (commitment, hint)
}

fn max_num_vars(program: &CommitmentCpuProgram) -> usize {
    program
        .batch_plans
        .iter()
        .map(|plan| plan.num_vars)
        .chain(program.optional_plans.iter().map(|plan| plan.num_vars))
        .max()
        .unwrap_or(0)
}

fn target_len(num_vars: usize) -> usize {
    1usize << num_vars
}

fn static_transcript_label(label: &str) -> &'static [u8] {
    match label {
        "commitment" => b"commitment",
        "untrusted_advice" => b"untrusted_advice",
        "trusted_advice" => b"trusted_advice",
        _ => panic!("unsupported transcript label `{label}`"),
    }
}
