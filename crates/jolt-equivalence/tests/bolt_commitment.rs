//! Commitment-phase transcript bridge between Bolt IR and jolt-core.
//!
//! This keeps the first oracle intentionally narrow: Bolt owns the commitment
//! ordering through its CPU IR, while jolt-core owns the reference
//! `append_serializable` transcript semantics for the same Dory commitments.

use std::{borrow::Cow, collections::BTreeMap};

use ark_serialize::CanonicalSerialize;
use bolt::{
    build_commitment_protocol, build_stage1_outer_protocol, build_stage2_protocol,
    build_stage3_protocol, build_stage4_protocol, build_stage5_protocol, build_stage6_protocol,
    build_stage7_protocol, build_stage8_protocol, commitment_cpu_program,
    lower_commitment_to_compute, lower_compute_to_cpu, lower_piop_and_fiat_shamir,
    lower_stage1_to_compute, lower_stage2_to_compute, lower_stage3_to_compute,
    lower_stage4_to_compute, lower_stage5_to_compute, lower_stage6_to_compute,
    lower_stage7_to_compute, lower_stage8_to_compute, project_prover_party, project_verifier_party,
    resolve_compute_kernels, stage1_cpu_program, stage2_cpu_program, stage3_cpu_program,
    stage4_cpu_program, stage5_cpu_program, stage6_cpu_program, stage7_cpu_program,
    stage8_cpu_program, CommitmentCpuProgram, JoltProtocolParams, MeliorContext,
    OptionalSkipPolicy, OracleGeneration, Role, TranscriptStep,
};
use bolt::{
    Stage1CpuProgram as CompilerStage1CpuProgram, Stage2CpuProgram as CompilerStage2CpuProgram,
    Stage3CpuProgram as CompilerStage3CpuProgram, Stage4CpuProgram as CompilerStage4CpuProgram,
    Stage5CpuProgram as CompilerStage5CpuProgram, Stage6CpuProgram as CompilerStage6CpuProgram,
    Stage7CpuProgram as CompilerStage7CpuProgram, Stage8CpuProgram as CompilerStage8CpuProgram,
};
use common::constants::{RAM_START_ADDRESS, XLEN};
use common::jolt_device::JoltDevice;
use jolt_core::curve::Bn254Curve;
use jolt_core::host;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme as CoreCommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::univariate_skip::{
    UniSkipFirstRoundProof, UniSkipFirstRoundProofVariant,
};
use jolt_core::transcripts::{Blake2bTranscript as CoreBlake2bTranscript, Transcript as _};
use jolt_core::zkvm::instruction::{
    CircuitFlags, Flags as CoreFlags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
    LookupQuery,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_core::zkvm::proof_serialization::{Claims as CoreClaims, JoltProof as CoreJoltProof};
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::r1cs::inputs::{
    ProductCycleInputs as CoreProductCycleInputs, R1CSCycleInputs as CoreR1CSCycleInputs,
};
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::spartan::verify_stage2_uni_skip;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use jolt_dory::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryScheme};
use jolt_equivalence::checkpoint::{
    assert_transcripts_match, CheckpointTranscript, TranscriptEvent,
};
use jolt_equivalence::core_conversion::{commitment_to_ark, to_ark, to_core_sumcheck_proof};
use jolt_field::signed::{S128, S64};
use jolt_field::{Field, Fr};
use jolt_kernels::stage1::{
    execute_stage1_program, Stage1CpuProgramPlan as KernelStage1CpuProgramPlan,
    Stage1ExecutionArtifacts, Stage1ExecutionMode, Stage1KernelPlan as KernelStage1KernelPlan,
    Stage1OpeningBatchPlan as KernelStage1OpeningBatchPlan,
    Stage1OpeningClaimPlan as KernelStage1OpeningClaimPlan, Stage1OuterR1csData,
    Stage1OuterRemainingEvaluator, Stage1OuterRv64Data, Stage1Params, Stage1Proof,
    Stage1ProverInputs, Stage1ProverKernelExecutor, Stage1Rv64Cycle,
    Stage1SumcheckBatchPlan as KernelStage1SumcheckBatchPlan,
    Stage1SumcheckClaimPlan as KernelStage1SumcheckClaimPlan,
    Stage1SumcheckDriverPlan as KernelStage1SumcheckDriverPlan,
    Stage1SumcheckEvalPlan as KernelStage1SumcheckEvalPlan,
    Stage1SumcheckInstanceResultPlan as KernelStage1SumcheckInstanceResultPlan,
    Stage1TranscriptSqueezePlan, Stage1VerifierKernelExecutor,
};
use jolt_kernels::stage2::{
    execute_stage2_program, product_virtual_uniskip_extended_evals,
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
use jolt_kernels::stage3::{
    execute_stage3_program, Stage3CpuProgramPlan as KernelStage3CpuProgramPlan, Stage3Cycle,
    Stage3ExecutionArtifacts, Stage3ExecutionMode,
    Stage3FieldConstantPlan as KernelStage3FieldConstantPlan,
    Stage3FieldExprPlan as KernelStage3FieldExprPlan, Stage3KernelPlan as KernelStage3KernelPlan,
    Stage3OpeningBatchPlan as KernelStage3OpeningBatchPlan,
    Stage3OpeningClaimEqualityPlan as KernelStage3OpeningClaimEqualityPlan,
    Stage3OpeningClaimPlan as KernelStage3OpeningClaimPlan, Stage3OpeningInputPlan,
    Stage3OpeningInputValue, Stage3Params, Stage3PointConcatPlan, Stage3PointSlicePlan,
    Stage3ProgramStepPlan as KernelStage3ProgramStepPlan, Stage3Proof, Stage3ProverInputs,
    Stage3ProverKernelExecutor, Stage3SumcheckBatchPlan as KernelStage3SumcheckBatchPlan,
    Stage3SumcheckClaimPlan as KernelStage3SumcheckClaimPlan,
    Stage3SumcheckDriverPlan as KernelStage3SumcheckDriverPlan,
    Stage3SumcheckEvalPlan as KernelStage3SumcheckEvalPlan, Stage3SumcheckInstanceResultPlan,
    Stage3TranscriptSqueezePlan as KernelStage3TranscriptSqueezePlan, Stage3VerifierKernelExecutor,
};
use jolt_kernels::stage4::{
    execute_stage4_program, Stage4CpuProgramPlan as KernelStage4CpuProgramPlan,
    Stage4ExecutionArtifacts, Stage4ExecutionMode,
    Stage4FieldConstantPlan as KernelStage4FieldConstantPlan,
    Stage4FieldExprPlan as KernelStage4FieldExprPlan, Stage4KernelPlan as KernelStage4KernelPlan,
    Stage4NamedEval, Stage4OpeningBatchPlan as KernelStage4OpeningBatchPlan,
    Stage4OpeningClaimEqualityPlan as KernelStage4OpeningClaimEqualityPlan,
    Stage4OpeningClaimPlan as KernelStage4OpeningClaimPlan, Stage4OpeningInputPlan,
    Stage4OpeningInputValue, Stage4Params, Stage4PointConcatPlan, Stage4PointSlicePlan,
    Stage4ProgramStepPlan as KernelStage4ProgramStepPlan, Stage4Proof, Stage4ProverInputs,
    Stage4ProverKernelExecutor, Stage4RamWitness, Stage4RegisterAccess, Stage4RegisterRead,
    Stage4RegisterWrite, Stage4RegistersWitness,
    Stage4SumcheckBatchPlan as KernelStage4SumcheckBatchPlan,
    Stage4SumcheckClaimPlan as KernelStage4SumcheckClaimPlan,
    Stage4SumcheckDriverPlan as KernelStage4SumcheckDriverPlan,
    Stage4SumcheckEvalPlan as KernelStage4SumcheckEvalPlan, Stage4SumcheckInstanceResultPlan,
    Stage4SumcheckOutput, Stage4TranscriptAbsorbBytesPlan as KernelStage4TranscriptAbsorbBytesPlan,
    Stage4TranscriptSqueezePlan as KernelStage4TranscriptSqueezePlan, Stage4VerifierKernelExecutor,
};
use jolt_kernels::stage5 as kernel_stage5;
use jolt_kernels::stage6 as kernel_stage6;
use jolt_openings::{CommitmentScheme as _, StreamingCommitment};
use jolt_poly::lagrange::lagrange_kernel_eval;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_prover::stages::{
    commitment as generated_prover_commitment, stage8 as generated_prover_stage8,
};
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_trace::{
    extract_trace, BytecodePreprocessing, CircuitFlags as TraceCircuitFlags, CycleRow,
    InstructionFlags as TraceInstructionFlags,
};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use jolt_verifier::stages::{
    commitment as generated_commitment, stage1_outer as generated_stage1,
    stage2 as generated_stage2, stage3 as generated_stage3, stage4 as generated_stage4,
    stage5 as generated_stage5, stage6 as generated_stage6, stage7 as generated_stage7,
    stage8 as generated_stage8,
};
use jolt_witness::{
    dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle, CycleInput,
};
use strum::EnumCount;
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

const TRANSCRIPT_LABEL: &[u8] = b"Jolt";

// Keep older monolithic verifier gates scoped to the stage prefix under test.
fn empty_generated_stage6_verifier_program() -> &'static generated_stage6::Stage6VerifierProgramPlan
{
    Box::leak(Box::new(generated_stage6::Stage6CpuProgramPlan {
        role: "verifier",
        params: generated_stage6::STAGE6_PARAMS,
        steps: &[],
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: &[],
        field_exprs: &[],
        kernels: &[],
        claims: &[],
        batches: &[],
        drivers: &[],
        instance_results: &[],
        evals: &[],
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    }))
}

fn empty_generated_stage7_verifier_program() -> &'static generated_stage7::Stage7VerifierProgramPlan
{
    Box::leak(Box::new(generated_stage7::Stage7CpuProgramPlan {
        role: "verifier",
        params: generated_stage7::STAGE7_PARAMS,
        steps: &[],
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: &[],
        field_exprs: &[],
        kernels: &[],
        claims: &[],
        batches: &[],
        drivers: &[],
        instance_results: &[],
        evals: &[],
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    }))
}

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
    let generated_stage1_verifier_plan =
        leak_generated_stage1_verifier_program(&stage1_verifier_program);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    assert_rv64_cycles_match_core(&fixture.rv64_cycles, &fixture.core_rv64_cycles);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid RV64-backed stage1 data");
    let core_row_data =
        Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.core_rv64_cycles)
            .expect("valid core RV64-backed stage1 data");
    let generic_data = Stage1OuterR1csData::new(&r1cs_key, &fixture.r1cs_witness)
        .expect("valid R1CS witness shape");

    let mut prover_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut prover_transcript, &fixture);
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
    assert_stage1_uniskip_extended_evals_match_core(
        &fixture.proof,
        &data,
        &core_row_data,
        &generic_data,
        &stage1_artifacts,
    );

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let mut verifier_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut verifier_transcript, &fixture);
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
    assert_core_preamble_states_match_bolt(&fixture, prover_transcript.log());

    assert_core_accepts_bolt_stage1(&fixture, &stage1_artifacts);
    assert_core_states_match_bolt_stage1(&fixture, prover_transcript.log());
    assert_bolt_stage1_tamper_rejected(
        stage1_verifier_plan,
        generated_stage1_verifier_plan,
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
    let (stage2_prover_program, stage2_verifier_program) =
        bolt_stage2_programs_with_params(&fixture.params);
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
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid RV64-backed stage1 data");

    let mut bolt_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut bolt_transcript, &fixture);
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
    let generated_stage2_product_verifier_plan =
        leak_generated_stage2_product_uniskip_verifier_program(&stage2_verifier_program);
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
        generated_stage2_verifier_plan: generated_stage2_product_verifier_plan,
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
    let (stage2_prover_program, stage2_verifier_program) =
        bolt_stage2_programs_with_params(&fixture.params);
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
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid RV64-backed stage1 data");

    let mut prover_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut prover_transcript, &fixture);
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
    let generated_stage2_verifier_plan =
        leak_generated_stage2_verifier_program(&stage2_verifier_program);
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
    append_bolt_preamble(&mut verifier_transcript, &fixture);
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

    let assert_stage2_tamper_rejected = |tampered_stage2_artifacts: Stage2ExecutionArtifacts<
        Fr,
    >,
                                         message: &str| {
        let mut tamper_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut tamper_transcript, &fixture);
        append_bolt_commitments_to_transcript(
            &mut tamper_transcript,
            &commitment_verifier_trace.records,
            &commitment_verifier_trace.commitments,
            &commitment_verifier_program.transcript_steps,
        );
        let mut tamper_stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
        let tamper_verified_stage1 = execute_stage1_program(
            stage1_prover_plan,
            Stage1ExecutionMode::Verifier,
            &mut tamper_stage1_verifier,
            &mut tamper_transcript,
        )
        .expect("Bolt Stage 1 verifier accepts before Stage 2 tamper");
        let tamper_stage2_openings = stage2_opening_inputs(&tamper_verified_stage1);
        let tampered_stage2_proof = Stage2Proof::from(tampered_stage2_artifacts.clone());
        let mut tamper_stage2_verifier =
            Stage2VerifierKernelExecutor::new(&tampered_stage2_proof, &tamper_stage2_openings)
                .with_ram_data(&ram_data);
        let tamper_result = execute_stage2_program(
            stage2_prover_plan,
            Stage2ExecutionMode::Verifier,
            &mut tamper_stage2_verifier,
            &mut tamper_transcript,
        );
        assert!(tamper_result.is_err(), "{message}");

        let mut generated_tamper_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut generated_tamper_transcript, &fixture);
        append_bolt_commitments_to_transcript(
            &mut generated_tamper_transcript,
            &commitment_verifier_trace.records,
            &commitment_verifier_trace.commitments,
            &commitment_verifier_program.transcript_steps,
        );
        let mut generated_tamper_stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
        let generated_tamper_verified_stage1 = execute_stage1_program(
            stage1_prover_plan,
            Stage1ExecutionMode::Verifier,
            &mut generated_tamper_stage1_verifier,
            &mut generated_tamper_transcript,
        )
        .expect("Bolt Stage 1 verifier accepts before generated Stage 2 tamper");
        let generated_tamper_stage2_openings = generated_stage2_opening_inputs(
            &stage2_opening_inputs(&generated_tamper_verified_stage1),
        );
        let generated_ram_accesses = generated_stage2_ram_accesses(ram_data.accesses);
        let generated_ram_layout = generated_stage2_ram_output_layout(fixture.ram_output_layout);
        let generated_ram_data = generated_stage2::Stage2RamData {
            log_k: ram_data.log_k,
            start_address: ram_data.start_address,
            initial_ram: ram_data.initial_ram,
            final_ram: ram_data.final_ram,
            accesses: &generated_ram_accesses,
            output_layout: Some(generated_ram_layout),
        };
        let generated_tampered_stage2_proof = to_generated_stage2_proof(&tampered_stage2_artifacts);
        let generated_tamper_result = generated_stage2::verify_stage2_with_program(
            generated_stage2_verifier_plan,
            &generated_tampered_stage2_proof,
            &generated_tamper_stage2_openings,
            Some(&generated_ram_data),
            &mut generated_tamper_transcript,
        );
        assert!(generated_tamper_result.is_err(), "generated {message}");
    };

    let mut tampered_sumcheck = stage2_artifacts.clone();
    let tampered_poly = &mut tampered_sumcheck.sumchecks[1].proof.round_polynomials[0];
    let mut tampered_coefficients = tampered_poly.clone().into_coefficients();
    tampered_coefficients[0] += Fr::from_u64(1);
    *tampered_poly = UnivariatePoly::new(tampered_coefficients);
    assert_stage2_tamper_rejected(
        tampered_sumcheck,
        "Bolt Stage 2 verifier accepted a tampered batched sumcheck coefficient",
    );

    let mut tampered_eval = stage2_artifacts.clone();
    tampered_eval.sumchecks[1].evals[0].value += Fr::from_u64(1);
    assert_stage2_tamper_rejected(
        tampered_eval,
        "Bolt Stage 2 verifier accepted a tampered batched opening evaluation",
    );

    let mut tampered_point = stage2_artifacts.clone();
    tampered_point.sumchecks[1].point[0] += Fr::from_u64(1);
    assert_stage2_tamper_rejected(
        tampered_point,
        "Bolt Stage 2 verifier accepted a tampered batched sumcheck point",
    );
}

#[test]
fn bolt_stage3_batched_real_muldiv_self_parity() {
    let fixture = core_muldiv_commitment_fixture();
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, stage1_verifier_program) =
        bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, stage2_verifier_program) =
        bolt_stage2_programs_with_params(&fixture.params);
    let (stage3_prover_program, stage3_verifier_program) =
        bolt_stage3_programs_with_params(&fixture.params);
    let (stage4_prover_program, stage4_verifier_program) =
        bolt_stage4_programs_with_params(&fixture.params);
    let (stage5_prover_program, stage5_verifier_program) =
        bolt_stage5_programs_with_params(&fixture.params);
    let (stage6_prover_program, stage6_verifier_program) =
        bolt_stage6_programs_with_params(&fixture.params);
    let (stage7_prover_program, stage7_verifier_program) =
        bolt_stage7_programs_with_params(&fixture.params);
    let (stage8_prover_program, stage8_verifier_program) =
        bolt_stage8_programs_with_params(&fixture.params);
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
    let stage2_prover_plan = leak_stage2_program(&stage2_prover_program);
    let stage3_prover_plan = leak_stage3_program(&stage3_prover_program);
    let stage4_prover_plan = leak_stage4_program(&stage4_prover_program);
    let stage5_prover_plan = leak_stage5_program(&stage5_prover_program);
    let stage6_prover_plan = leak_stage6_program(&stage6_prover_program);
    let stage7_prover_plan = leak_stage7_program(&stage7_prover_program);
    let stage8_prover_plan = leak_generated_stage8_prover_program(&stage8_prover_program);
    let generated_commitment_prover_plan =
        leak_generated_commitment_prover_program(&commitment_prover_program);
    let generated_commitment_verifier_plan =
        leak_generated_commitment_verifier_program(&commitment_verifier_program);
    let generated_stage1_verifier_plan =
        leak_generated_stage1_verifier_program(&stage1_verifier_program);
    let generated_stage2_verifier_plan =
        leak_generated_stage2_verifier_program(&stage2_verifier_program);
    let generated_stage3_verifier_plan =
        leak_generated_stage3_verifier_program(&stage3_verifier_program);
    let generated_stage4_verifier_plan =
        leak_generated_stage4_verifier_program(&stage4_verifier_program);
    let generated_stage5_verifier_plan =
        leak_generated_stage5_verifier_program(&stage5_verifier_program);
    let generated_stage6_verifier_plan =
        leak_generated_stage6_verifier_program(&stage6_verifier_program);
    let generated_stage7_verifier_plan =
        leak_generated_stage7_verifier_program(&stage7_verifier_program);
    let generated_stage8_verifier_plan =
        leak_generated_stage8_verifier_program(&stage8_verifier_program);
    let generated_programs = jolt_verifier::JoltVerifierPrograms {
        commitment: generated_commitment_verifier_plan,
        stage1_outer: generated_stage1_verifier_plan,
        stage2: generated_stage2_verifier_plan,
        stage3: generated_stage3_verifier_plan,
        stage4: generated_stage4_verifier_plan,
        stage5: generated_stage5_verifier_plan,
        stage6: empty_generated_stage6_verifier_program(),
        stage7: empty_generated_stage7_verifier_program(),
        stage8: generated_stage8_verifier_plan,
    };
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid RV64-backed stage1 data");

    let mut prover_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut prover_transcript, &fixture);
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
    let mut stage2_prover = Stage2ProverKernelExecutor::new(stage2_inputs);
    let stage2_artifacts = execute_stage2_program(
        stage2_prover_plan,
        Stage2ExecutionMode::Prover,
        &mut stage2_prover,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 2 prover succeeds");

    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let stage3_inputs =
        Stage3ProverInputs::new(&stage3_openings).with_cycles(&fixture.stage3_cycles);
    let mut stage3_prover = Stage3ProverKernelExecutor::new(stage3_inputs);
    let stage3_artifacts = execute_stage3_program(
        stage3_prover_plan,
        Stage3ExecutionMode::Prover,
        &mut stage3_prover,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 3 prover succeeds");

    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let core_stage4 = core_stage4_artifacts(&fixture);
    assert_stage4_opening_inputs_match(&core_stage4.opening_inputs, &stage4_openings);
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_registers = Stage4RegistersWitness {
        register_count: 1 << fixture.params.register_log_k,
        trace_len: fixture.proof.trace_length,
        registers_val: &[],
        rs1_ra: &[],
        rs2_ra: &[],
        rd_wa: &[],
        accesses: Some(&fixture.stage4_register_accesses),
        rd_inc: &stage4_rd_inc,
    };
    let stage4_ram = Stage4RamWitness {
        ram_k: fixture.proof.ram_K,
        trace_len: fixture.proof.trace_length,
        ram_ra: &[],
        write_address_indices: Some(&stage4_ram_addresses),
        ram_inc: &stage4_ram_inc,
    };
    let stage4_inputs = Stage4ProverInputs::new(&stage4_openings)
        .with_registers(stage4_registers)
        .with_ram(stage4_ram);
    let mut stage4_prover = Stage4ProverKernelExecutor::new(stage4_inputs);
    let stage4_artifacts = execute_stage4_program(
        stage4_prover_plan,
        Stage4ExecutionMode::Prover,
        &mut stage4_prover,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 4 prover succeeds");

    let mut verifier_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut verifier_transcript, &fixture);
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
    let stage3_verifier_openings = stage3_opening_inputs(&verified_stage1, &verified_stage2);
    let stage3_proof = Stage3Proof::from(stage3_artifacts.clone());
    let mut stage3_verifier =
        Stage3VerifierKernelExecutor::new(&stage3_proof, &stage3_verifier_openings);
    let verified_stage3 = execute_stage3_program(
        stage3_prover_plan,
        Stage3ExecutionMode::Verifier,
        &mut stage3_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 3 verifier accepts");
    let stage4_verifier_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &verified_stage2,
        &verified_stage3,
    );
    let stage4_proof = Stage4Proof::from(stage4_artifacts.clone());
    let mut stage4_verifier =
        Stage4VerifierKernelExecutor::new(&stage4_proof, &stage4_verifier_openings);
    let verified_stage4 = execute_stage4_program(
        stage4_prover_plan,
        Stage4ExecutionMode::Verifier,
        &mut stage4_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 4 verifier accepts");

    assert_eq!(
        stage3_artifacts.sumchecks.len(),
        verified_stage3.sumchecks.len()
    );
    assert_eq!(
        stage4_artifacts.sumchecks.len(),
        verified_stage4.sumchecks.len()
    );
    assert_state_history_match(
        &transcript_states(prover_transcript.log()),
        &transcript_states(verifier_transcript.log()),
    );
    assert_core_accepts_bolt_stage3(
        &fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    assert_core_accepts_bolt_stage4(
        &fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
    );

    assert_stage4_artifacts_match(&core_stage4.artifacts, &stage4_artifacts);
    let core_stage4_states = core_stage4.transcript_states;
    assert_state_history_match(
        &core_stage4_states,
        &transcript_states(prover_transcript.log()),
    );

    let mut generated_verifier_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut generated_verifier_transcript, &fixture);
    append_bolt_commitments_to_transcript(
        &mut generated_verifier_transcript,
        &commitment_verifier_trace.records,
        &commitment_verifier_trace.commitments,
        &commitment_verifier_program.transcript_steps,
    );
    let mut generated_stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
    let generated_verified_stage1 = execute_stage1_program(
        stage1_prover_plan,
        Stage1ExecutionMode::Verifier,
        &mut generated_stage1_verifier,
        &mut generated_verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts before generated Stage 3");
    let generated_stage2_openings = stage2_opening_inputs(&generated_verified_stage1);
    let mut generated_stage2_verifier =
        Stage2VerifierKernelExecutor::new(&stage2_proof, &generated_stage2_openings)
            .with_ram_data(&ram_data);
    let generated_verified_stage2 = execute_stage2_program(
        stage2_prover_plan,
        Stage2ExecutionMode::Verifier,
        &mut generated_stage2_verifier,
        &mut generated_verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts before generated Stage 3");
    let generated_stage3_openings = generated_stage3_opening_inputs(&stage3_opening_inputs(
        &generated_verified_stage1,
        &generated_verified_stage2,
    ));
    let generated_stage3_proof = to_generated_stage3_proof(&stage3_artifacts);
    let generated_verified_stage3 = generated_stage3::verify_stage3_with_program(
        generated_stage3_verifier_plan,
        &generated_stage3_proof,
        &generated_stage3_openings,
        &mut generated_verifier_transcript,
    )
    .expect("generated Stage 3 verifier accepts real muldiv proof");
    assert_eq!(
        stage3_artifacts.sumchecks.len(),
        generated_verified_stage3.sumchecks.len()
    );
    let generated_stage4_openings = generated_stage4_opening_inputs(&stage4_verifier_openings);
    let generated_stage4_proof = to_generated_stage4_proof(&stage4_artifacts);
    let generated_verified_stage4 = generated_stage4::verify_stage4_with_program(
        generated_stage4_verifier_plan,
        &generated_stage4_proof,
        &generated_stage4_openings,
        &mut generated_verifier_transcript,
    )
    .expect("generated Stage 4 verifier accepts real muldiv proof");
    assert_eq!(
        stage4_artifacts.sumchecks.len(),
        generated_verified_stage4.sumchecks.len()
    );
    assert_state_history_match(
        &core_stage4_states,
        &transcript_states(generated_verifier_transcript.log()),
    );
    let core_stage5 = core_stage5_artifacts(&fixture);
    let generated_stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    assert_stage5_opening_inputs_match(&core_stage5.opening_inputs, &generated_stage5_openings);
    let kernel_stage5_openings = kernel_stage5_opening_inputs(&generated_stage5_openings);
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let stage5_inputs = kernel_stage5::Stage5ProverInputs::new(&kernel_stage5_openings)
        .with_instruction_read_raf(kernel_stage5::Stage5InstructionReadRafWitness {
            trace_len: fixture.proof.trace_length,
            lookup_indices: &fixture.stage5_lookup_indices,
            lookup_table_indices: &fixture.stage5_lookup_table_indices,
            is_interleaved_operands: &fixture.stage5_is_interleaved_operands,
            ra_virtual_log_k_chunk: fixture.params.lookups_ra_virtual_log_k_chunk,
        })
        .with_ram_ra(kernel_stage5::Stage5RamRaWitness {
            ram_k: fixture.proof.ram_K,
            trace_len: fixture.proof.trace_length,
            ram_ra: &[],
            remapped_addresses: Some(&stage4_ram_addresses),
        })
        .with_registers_val(kernel_stage5::Stage5RegistersValWitness {
            register_count: 1 << fixture.params.register_log_k,
            trace_len: fixture.proof.trace_length,
            rd_inc: &stage4_rd_inc,
            rd_wa: &[],
            rd_write_addresses: Some(&stage5_rd_write_addresses),
        });
    let mut stage5_prover_transcript = prover_transcript.clone();
    let mut stage5_prover = kernel_stage5::Stage5ProverKernelExecutor::new(stage5_inputs);
    let stage5_artifacts = kernel_stage5::execute_stage5_program(
        stage5_prover_plan,
        kernel_stage5::Stage5ExecutionMode::Prover,
        &mut stage5_prover,
        &mut stage5_prover_transcript,
    )
    .expect("Bolt Stage 5 prover succeeds");
    assert_eq!(stage5_artifacts.sumchecks.len(), 1);
    assert_state_history_match(
        &core_stage5.transcript_states,
        &transcript_states(stage5_prover_transcript.log()),
    );
    let stage5_proof = kernel_stage5::Stage5Proof {
        sumchecks: stage5_artifacts.sumchecks.clone(),
    };
    let mut kernel_stage5_transcript = verifier_transcript.clone();
    let mut kernel_stage5_executor = kernel_stage5::Stage5ProofCarryingKernelExecutor::new(
        &stage5_proof,
        &kernel_stage5_openings,
    );
    let kernel_verified_stage5 = kernel_stage5::execute_stage5_program(
        stage5_prover_plan,
        kernel_stage5::Stage5ExecutionMode::Prover,
        &mut kernel_stage5_executor,
        &mut kernel_stage5_transcript,
    )
    .expect("kernel Stage 5 replay accepts Bolt real muldiv proof");
    assert_eq!(kernel_verified_stage5.sumchecks.len(), 1);
    assert_state_history_match(
        &core_stage5.transcript_states,
        &transcript_states(kernel_stage5_transcript.log()),
    );
    let generated_stage5_proof = to_generated_stage5_proof(&stage5_artifacts);
    assert_stage5_artifacts_match(&core_stage5.proof, &generated_stage5_proof);
    let generated_stage5_start_transcript = generated_verifier_transcript.clone();
    let generated_verified_stage5 = generated_stage5::verify_stage5_with_program(
        generated_stage5_verifier_plan,
        &generated_stage5_proof,
        &generated_stage5_openings,
        &mut generated_verifier_transcript,
    )
    .expect("generated Stage 5 verifier accepts Bolt real muldiv proof");
    assert_eq!(generated_verified_stage5.sumchecks.len(), 1);
    assert_state_history_match(
        &core_stage5.transcript_states,
        &transcript_states(generated_verifier_transcript.log()),
    );
    let core_stage6 = core_stage6_artifacts(&fixture);
    let generated_stage6_data = generated_stage6_verifier_data(&fixture);
    let kernel_stage6_proof = to_kernel_stage6_proof(&core_stage6.proof);
    let kernel_stage6_openings = kernel_stage6_opening_inputs(&core_stage6.opening_inputs);
    let kernel_stage6_bytecode_entries = kernel_stage6_bytecode_entries(
        &generated_stage6_data
            .bytecode_read_raf
            .as_ref()
            .expect("Stage 6 bytecode verifier data")
            .entries,
    );
    let kernel_stage6_bytecode_data = kernel_stage6::Stage6BytecodeReadRafData {
        entries: &kernel_stage6_bytecode_entries,
        entry_bytecode_index: generated_stage6_data
            .bytecode_read_raf
            .as_ref()
            .expect("Stage 6 bytecode verifier data")
            .entry_bytecode_index,
        num_lookup_tables: generated_stage6_data
            .bytecode_read_raf
            .as_ref()
            .expect("Stage 6 bytecode verifier data")
            .num_lookup_tables,
    };
    let stage6_witness = stage6_witness_polynomials(&fixture, &kernel_stage6_openings);
    let mut stage6_booleanity_chunks = Vec::new();
    stage6_booleanity_chunks.extend(
        stage6_witness
            .instruction_ra_booleanity
            .iter()
            .map(Vec::as_slice),
    );
    stage6_booleanity_chunks.extend(
        stage6_witness
            .bytecode_ra_booleanity
            .iter()
            .map(Vec::as_slice),
    );
    stage6_booleanity_chunks.extend(stage6_witness.ram_ra_booleanity.iter().map(Vec::as_slice));
    let stage6_bytecode_ra_chunks = stage6_witness
        .bytecode_ra_read_raf
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let stage6_ram_ra_chunks = stage6_witness
        .ram_ra_virtual
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let stage6_instruction_ra_chunks = stage6_witness
        .instruction_ra_virtual
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let stage6_inputs = kernel_stage6::Stage6ProverInputs::new(&kernel_stage6_openings)
        .with_bytecode_read_raf(kernel_stage6::Stage6BytecodeReadRafWitness {
            data: kernel_stage6_bytecode_data,
            bytecode_ra_chunks: &stage6_bytecode_ra_chunks,
        })
        .with_booleanity(kernel_stage6::Stage6BooleanityWitness {
            chunks: &stage6_booleanity_chunks,
        })
        .with_hamming_booleanity(kernel_stage6::Stage6HammingBooleanityWitness {
            hamming_weight: &stage6_witness.hamming_weight,
        })
        .with_ram_ra_virtual(kernel_stage6::Stage6RamRaVirtualWitness {
            ram_ra_chunks: &stage6_ram_ra_chunks,
        })
        .with_instruction_ra_virtual(kernel_stage6::Stage6InstructionRaVirtualWitness {
            instruction_ra_chunks: &stage6_instruction_ra_chunks,
            virtual_count: fixture.params.instruction_ra_virtual_d,
        })
        .with_inc_claim_reduction(kernel_stage6::Stage6IncClaimReductionWitness {
            ram_inc: &stage6_witness.ram_inc,
            rd_inc: &stage6_witness.rd_inc,
        });
    let mut stage6_prover_transcript = generated_verifier_transcript.clone();
    let mut stage6_prover = kernel_stage6::Stage6ProverKernelExecutor::new(stage6_inputs);
    let stage6_artifacts = kernel_stage6::execute_stage6_program(
        stage6_prover_plan,
        kernel_stage6::Stage6ExecutionMode::Prover,
        &mut stage6_prover,
        &mut stage6_prover_transcript,
    )
    .expect("Bolt Stage 6 prover succeeds");
    assert_eq!(stage6_artifacts.sumchecks.len(), 1);
    let generated_stage6_artifacts = generated_stage6_execution_artifacts(&stage6_artifacts);
    let generated_stage6_proof = generated_stage6::Stage6Proof {
        sumchecks: generated_stage6_artifacts.sumchecks.clone(),
    };
    assert_stage6_artifacts_match(&core_stage6.proof, &generated_stage6_artifacts);
    assert_state_history_match(
        &core_stage6.transcript_states,
        &transcript_states(stage6_prover_transcript.log()),
    );
    let mut kernel_stage6_transcript = generated_verifier_transcript.clone();
    let mut kernel_stage6_executor = kernel_stage6::Stage6ProofCarryingKernelExecutor::new(
        &kernel_stage6_proof,
        &kernel_stage6_openings,
    )
    .with_bytecode_read_raf_data(kernel_stage6_bytecode_data);
    let kernel_verified_stage6 = kernel_stage6::execute_stage6_program(
        stage6_prover_plan,
        kernel_stage6::Stage6ExecutionMode::Prover,
        &mut kernel_stage6_executor,
        &mut kernel_stage6_transcript,
    )
    .expect("kernel Stage 6 replay accepts jolt-core real muldiv proof");
    assert_eq!(kernel_verified_stage6.sumchecks.len(), 1);
    assert_stage6_artifacts_match(
        &core_stage6.proof,
        &generated_stage6_execution_artifacts(&kernel_verified_stage6),
    );
    assert_state_history_match(
        &core_stage6.transcript_states,
        &transcript_states(kernel_stage6_transcript.log()),
    );
    let mut generated_stage6_transcript = generated_verifier_transcript.clone();
    let generated_verified_stage6 = generated_stage6::verify_stage6_with_program(
        generated_stage6_verifier_plan,
        &core_stage6.proof,
        &core_stage6.opening_inputs,
        Some(&generated_stage6_data),
        &mut generated_stage6_transcript,
    )
    .expect("generated Stage 6 verifier accepts jolt-core real muldiv proof");
    assert_stage6_artifacts_match(&core_stage6.proof, &generated_verified_stage6);
    assert_state_history_match(
        &core_stage6.transcript_states,
        &transcript_states(generated_stage6_transcript.log()),
    );
    let mut generated_bolt_stage6_transcript = generated_verifier_transcript.clone();
    let generated_verified_bolt_stage6 = generated_stage6::verify_stage6_with_program(
        generated_stage6_verifier_plan,
        &generated_stage6_proof,
        &core_stage6.opening_inputs,
        Some(&generated_stage6_data),
        &mut generated_bolt_stage6_transcript,
    )
    .expect("generated Stage 6 verifier accepts Bolt real muldiv proof");
    assert_stage6_artifacts_match(&generated_stage6_proof, &generated_verified_bolt_stage6);
    assert_state_history_match(
        &core_stage6.transcript_states,
        &transcript_states(generated_bolt_stage6_transcript.log()),
    );

    let core_stage7 = core_stage7_artifacts(&fixture);
    let kernel_stage7_proof = to_kernel_stage7_proof(&core_stage7.proof);
    let kernel_stage7_openings = kernel_stage7_opening_inputs(&core_stage7.opening_inputs);
    let stage7_instruction_ra_chunks = stage6_witness
        .instruction_ra_booleanity
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let stage7_bytecode_ra_chunks = stage6_witness
        .bytecode_ra_booleanity
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let stage7_ram_ra_chunks = stage6_witness
        .ram_ra_booleanity
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let stage7_inputs = jolt_kernels::stage7::Stage7ProverInputs::new(&kernel_stage7_openings)
        .with_hamming_weight_claim_reduction(
            jolt_kernels::stage7::Stage7HammingWeightClaimReductionWitness {
                instruction_ra: jolt_kernels::stage7::Stage7RaChunks {
                    chunks: &stage7_instruction_ra_chunks,
                    layout: jolt_kernels::stage7::Stage7RaChunkLayout::CycleMajor,
                },
                bytecode_ra: jolt_kernels::stage7::Stage7RaChunks {
                    chunks: &stage7_bytecode_ra_chunks,
                    layout: jolt_kernels::stage7::Stage7RaChunkLayout::CycleMajor,
                },
                ram_ra: jolt_kernels::stage7::Stage7RaChunks {
                    chunks: &stage7_ram_ra_chunks,
                    layout: jolt_kernels::stage7::Stage7RaChunkLayout::CycleMajor,
                },
            },
        );
    let mut stage7_prover_transcript = stage6_prover_transcript.clone();
    let mut stage7_prover = jolt_kernels::stage7::Stage7ProverKernelExecutor::new(stage7_inputs);
    let stage7_artifacts = jolt_kernels::stage7::execute_stage7_program(
        stage7_prover_plan,
        jolt_kernels::stage7::Stage7ExecutionMode::Prover,
        &mut stage7_prover,
        &mut stage7_prover_transcript,
    )
    .expect("Bolt Stage 7 prover succeeds");
    assert_eq!(stage7_artifacts.sumchecks.len(), 1);
    let generated_stage7_artifacts = generated_stage7_execution_artifacts(&stage7_artifacts);
    let generated_stage7_proof = generated_stage7::Stage7Proof {
        sumchecks: generated_stage7_artifacts.sumchecks.clone(),
    };
    assert_stage7_artifacts_match(&core_stage7.proof, &generated_stage7_artifacts);
    assert_state_history_match(
        &core_stage7.transcript_states,
        &transcript_states(stage7_prover_transcript.log()),
    );
    let mut kernel_stage7_transcript = generated_stage6_transcript.clone();
    let mut kernel_stage7_executor = jolt_kernels::stage7::Stage7ProofCarryingKernelExecutor::new(
        &kernel_stage7_proof,
        &kernel_stage7_openings,
    );
    let kernel_verified_stage7 = jolt_kernels::stage7::execute_stage7_program(
        stage7_prover_plan,
        jolt_kernels::stage7::Stage7ExecutionMode::Prover,
        &mut kernel_stage7_executor,
        &mut kernel_stage7_transcript,
    )
    .expect("kernel Stage 7 replay accepts jolt-core real muldiv proof");
    assert_eq!(kernel_verified_stage7.sumchecks.len(), 1);
    assert_stage7_artifacts_match(
        &core_stage7.proof,
        &generated_stage7_execution_artifacts(&kernel_verified_stage7),
    );
    assert_state_history_match(
        &core_stage7.transcript_states,
        &transcript_states(kernel_stage7_transcript.log()),
    );
    let mut generated_stage7_transcript = generated_stage6_transcript.clone();
    let generated_verified_stage7 = generated_stage7::verify_stage7_with_program(
        generated_stage7_verifier_plan,
        &core_stage7.proof,
        &core_stage7.opening_inputs,
        &mut generated_stage7_transcript,
    )
    .expect("generated Stage 7 verifier accepts jolt-core real muldiv proof");
    assert_stage7_artifacts_match(&core_stage7.proof, &generated_verified_stage7);
    assert_state_history_match(
        &core_stage7.transcript_states,
        &transcript_states(generated_stage7_transcript.log()),
    );
    let mut generated_bolt_stage7_transcript = generated_bolt_stage6_transcript.clone();
    let generated_verified_bolt_stage7 = generated_stage7::verify_stage7_with_program(
        generated_stage7_verifier_plan,
        &generated_stage7_proof,
        &core_stage7.opening_inputs,
        &mut generated_bolt_stage7_transcript,
    )
    .expect("generated Stage 7 verifier accepts Bolt real muldiv proof");
    assert_stage7_artifacts_match(&generated_stage7_proof, &generated_verified_bolt_stage7);
    assert_state_history_match(
        &core_stage7.transcript_states,
        &transcript_states(generated_bolt_stage7_transcript.log()),
    );

    let generated_jolt_stage2_openings = generated_stage2_opening_inputs(&stage2_verifier_openings);
    let generated_jolt_stage3_openings = generated_stage3_opening_inputs(&stage3_verifier_openings);
    let generated_jolt_stage4_openings = generated_stage4_opening_inputs(&stage4_verifier_openings);
    let generated_jolt_stage5_openings = generated_stage5_openings.clone();
    let generated_ram_accesses = ram_data
        .accesses
        .iter()
        .map(|access| generated_stage2::Stage2RamAccess {
            remapped_address: access.remapped_address,
            read_value: access.read_value,
            write_value: access.write_value,
        })
        .collect::<Vec<_>>();
    let generated_ram_layout = generated_stage2::Stage2RamOutputLayout {
        io_start: fixture.ram_output_layout.io_start,
        io_end: fixture.ram_output_layout.io_end,
    };
    let generated_ram_data = generated_stage2::Stage2RamData {
        log_k: ram_data.log_k,
        start_address: ram_data.start_address,
        initial_ram: ram_data.initial_ram,
        final_ram: ram_data.final_ram,
        accesses: &generated_ram_accesses,
        output_layout: Some(generated_ram_layout),
    };
    let generated_jolt_proof = to_generated_jolt_proof(
        &commitment_verifier_trace.commitments,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
    );
    let mut generated_jolt_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut generated_jolt_transcript, &fixture);
    let generated_jolt_artifacts = jolt_verifier::verify_jolt_with_programs(
        &generated_jolt_proof,
        jolt_verifier::JoltVerifierInputs {
            stage2_openings: &generated_jolt_stage2_openings,
            stage2_ram: Some(&generated_ram_data),
            stage3_openings: &generated_jolt_stage3_openings,
            stage4_openings: &generated_jolt_stage4_openings,
            stage5_openings: &generated_jolt_stage5_openings,
            stage6_openings: &[],
            stage6_data: Some(&generated_stage6_data),
            stage7_openings: &[],
            evaluation_setup: None,
        },
        generated_programs,
        &mut generated_jolt_transcript,
    )
    .expect("generated monolithic verifier accepts real muldiv proof");
    assert_eq!(
        stage3_artifacts.sumchecks.len(),
        generated_jolt_artifacts.stage3.sumchecks.len()
    );
    assert_eq!(
        stage4_artifacts.sumchecks.len(),
        generated_jolt_artifacts.stage4.sumchecks.len()
    );
    assert_eq!(
        generated_stage5_proof.sumchecks.len(),
        generated_jolt_artifacts.stage5.sumchecks.len()
    );
    assert_state_history_match(
        &core_stage5.transcript_states,
        &transcript_states(generated_jolt_transcript.log()),
    );
    let generated_stage6_programs = jolt_verifier::JoltVerifierPrograms {
        stage6: generated_stage6_verifier_plan,
        ..generated_programs
    };
    let generated_jolt_proof_with_stage6 = to_generated_jolt_proof_with_stage6(
        &commitment_verifier_trace.commitments,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
        &generated_stage6_proof,
    );
    let mut generated_jolt_stage6_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut generated_jolt_stage6_transcript, &fixture);
    let generated_jolt_stage6_artifacts = jolt_verifier::verify_jolt_with_programs(
        &generated_jolt_proof_with_stage6,
        jolt_verifier::JoltVerifierInputs {
            stage2_openings: &generated_jolt_stage2_openings,
            stage2_ram: Some(&generated_ram_data),
            stage3_openings: &generated_jolt_stage3_openings,
            stage4_openings: &generated_jolt_stage4_openings,
            stage5_openings: &generated_jolt_stage5_openings,
            stage6_openings: &core_stage6.opening_inputs,
            stage6_data: Some(&generated_stage6_data),
            stage7_openings: &[],
            evaluation_setup: None,
        },
        generated_stage6_programs,
        &mut generated_jolt_stage6_transcript,
    )
    .expect("generated monolithic verifier accepts Bolt Stage 6 proof");
    assert_eq!(
        generated_stage6_proof.sumchecks.len(),
        generated_jolt_stage6_artifacts.stage6.sumchecks.len()
    );
    assert_state_history_match(
        &core_stage6.transcript_states,
        &transcript_states(generated_jolt_stage6_transcript.log()),
    );
    let generated_stage7_programs = jolt_verifier::JoltVerifierPrograms {
        stage6: generated_stage6_verifier_plan,
        stage7: generated_stage7_verifier_plan,
        ..generated_programs
    };
    let generated_jolt_proof_with_stage7 = to_generated_jolt_proof_with_stage7(
        &commitment_verifier_trace.commitments,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
        &generated_stage6_proof,
        &generated_stage7_proof,
    );
    let mut generated_jolt_stage7_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut generated_jolt_stage7_transcript, &fixture);
    let generated_jolt_stage7_artifacts = jolt_verifier::verify_jolt_with_programs(
        &generated_jolt_proof_with_stage7,
        jolt_verifier::JoltVerifierInputs {
            stage2_openings: &generated_jolt_stage2_openings,
            stage2_ram: Some(&generated_ram_data),
            stage3_openings: &generated_jolt_stage3_openings,
            stage4_openings: &generated_jolt_stage4_openings,
            stage5_openings: &generated_jolt_stage5_openings,
            stage6_openings: &core_stage6.opening_inputs,
            stage6_data: Some(&generated_stage6_data),
            stage7_openings: &core_stage7.opening_inputs,
            evaluation_setup: None,
        },
        generated_stage7_programs,
        &mut generated_jolt_stage7_transcript,
    )
    .expect("generated monolithic verifier accepts Bolt Stage 7 proof");
    assert_eq!(
        generated_stage7_proof.sumchecks.len(),
        generated_jolt_stage7_artifacts.stage7.sumchecks.len()
    );
    assert_state_history_match(
        &core_stage7.transcript_states,
        &transcript_states(generated_jolt_stage7_transcript.log()),
    );

    let monolithic_prover_programs = jolt_prover::JoltProverPrograms {
        commitment: generated_commitment_prover_plan,
        stage1_outer: stage1_prover_plan,
        stage2: stage2_prover_plan,
        stage3: stage3_prover_plan,
        stage4: stage4_prover_plan,
        stage5: stage5_prover_plan,
        stage6: stage6_prover_plan,
        stage7: stage7_prover_plan,
        stage8: stage8_prover_plan,
    };
    let mut monolithic_commitment_inputs = GeneratedCommitmentOracleMap {
        data: oracle_data.clone(),
    };
    let monolithic_stage1_inputs =
        Stage1ProverInputs::empty(r1cs_key.num_cycle_vars()).with_outer_remaining_evaluator(&data);
    let mut monolithic_stage1_prover = Stage1ProverKernelExecutor::new(monolithic_stage1_inputs);
    let monolithic_stage2_inputs = Stage2ProverInputs::new(&stage2_openings)
        .with_product_uniskip_extended_evals(&extended_evals)
        .with_product_virtual_cycles(&fixture.product_virtual_cycles)
        .with_instruction_lookup_cycles(&fixture.instruction_lookup_cycles)
        .with_ram_data(&ram_data);
    let mut monolithic_stage2_prover = Stage2ProverKernelExecutor::new(monolithic_stage2_inputs);
    let monolithic_stage3_inputs =
        Stage3ProverInputs::new(&stage3_openings).with_cycles(&fixture.stage3_cycles);
    let mut monolithic_stage3_prover = Stage3ProverKernelExecutor::new(monolithic_stage3_inputs);
    let monolithic_stage4_registers = Stage4RegistersWitness {
        register_count: 1 << fixture.params.register_log_k,
        trace_len: fixture.proof.trace_length,
        registers_val: &[],
        rs1_ra: &[],
        rs2_ra: &[],
        rd_wa: &[],
        accesses: Some(&fixture.stage4_register_accesses),
        rd_inc: &stage4_rd_inc,
    };
    let monolithic_stage4_ram = Stage4RamWitness {
        ram_k: fixture.proof.ram_K,
        trace_len: fixture.proof.trace_length,
        ram_ra: &[],
        write_address_indices: Some(&stage4_ram_addresses),
        ram_inc: &stage4_ram_inc,
    };
    let monolithic_stage4_inputs = Stage4ProverInputs::new(&stage4_openings)
        .with_registers(monolithic_stage4_registers)
        .with_ram(monolithic_stage4_ram);
    let mut monolithic_stage4_prover = Stage4ProverKernelExecutor::new(monolithic_stage4_inputs);
    let monolithic_stage5_inputs = kernel_stage5::Stage5ProverInputs::new(&kernel_stage5_openings)
        .with_instruction_read_raf(kernel_stage5::Stage5InstructionReadRafWitness {
            trace_len: fixture.proof.trace_length,
            lookup_indices: &fixture.stage5_lookup_indices,
            lookup_table_indices: &fixture.stage5_lookup_table_indices,
            is_interleaved_operands: &fixture.stage5_is_interleaved_operands,
            ra_virtual_log_k_chunk: fixture.params.lookups_ra_virtual_log_k_chunk,
        })
        .with_ram_ra(kernel_stage5::Stage5RamRaWitness {
            ram_k: fixture.proof.ram_K,
            trace_len: fixture.proof.trace_length,
            ram_ra: &[],
            remapped_addresses: Some(&stage4_ram_addresses),
        })
        .with_registers_val(kernel_stage5::Stage5RegistersValWitness {
            register_count: 1 << fixture.params.register_log_k,
            trace_len: fixture.proof.trace_length,
            rd_inc: &stage4_rd_inc,
            rd_wa: &[],
            rd_write_addresses: Some(&stage5_rd_write_addresses),
        });
    let mut monolithic_stage5_prover =
        kernel_stage5::Stage5ProverKernelExecutor::new(monolithic_stage5_inputs);
    let monolithic_stage6_inputs = kernel_stage6::Stage6ProverInputs::new(&kernel_stage6_openings)
        .with_bytecode_read_raf(kernel_stage6::Stage6BytecodeReadRafWitness {
            data: kernel_stage6_bytecode_data,
            bytecode_ra_chunks: &stage6_bytecode_ra_chunks,
        })
        .with_booleanity(kernel_stage6::Stage6BooleanityWitness {
            chunks: &stage6_booleanity_chunks,
        })
        .with_hamming_booleanity(kernel_stage6::Stage6HammingBooleanityWitness {
            hamming_weight: &stage6_witness.hamming_weight,
        })
        .with_ram_ra_virtual(kernel_stage6::Stage6RamRaVirtualWitness {
            ram_ra_chunks: &stage6_ram_ra_chunks,
        })
        .with_instruction_ra_virtual(kernel_stage6::Stage6InstructionRaVirtualWitness {
            instruction_ra_chunks: &stage6_instruction_ra_chunks,
            virtual_count: fixture.params.instruction_ra_virtual_d,
        })
        .with_inc_claim_reduction(kernel_stage6::Stage6IncClaimReductionWitness {
            ram_inc: &stage6_witness.ram_inc,
            rd_inc: &stage6_witness.rd_inc,
        });
    let mut monolithic_stage6_prover =
        kernel_stage6::Stage6ProverKernelExecutor::new(monolithic_stage6_inputs);
    let mut monolithic_stage7_prover =
        jolt_kernels::stage7::Stage7ProverKernelExecutor::new(stage7_inputs);
    let mut monolithic_prover_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut monolithic_prover_transcript, &fixture);
    let (monolithic_proof, monolithic_artifacts) = jolt_prover::prove_jolt_with_programs(
        jolt_prover::JoltProverInputs {
            commitment_inputs: &mut monolithic_commitment_inputs,
            prover_setup: &fixture.pcs_setup,
            stage1_outer_executor: &mut monolithic_stage1_prover,
            stage2_executor: &mut monolithic_stage2_prover,
            stage3_executor: &mut monolithic_stage3_prover,
            stage4_executor: &mut monolithic_stage4_prover,
            stage5_executor: &mut monolithic_stage5_prover,
            stage6_executor: &mut monolithic_stage6_prover,
            stage7_executor: &mut monolithic_stage7_prover,
            stage7_openings: Some(&kernel_stage7_openings),
        },
        monolithic_prover_programs,
        &mut monolithic_prover_transcript,
    )
    .expect("generated monolithic prover produces real muldiv proof");
    let monolithic_evaluation = monolithic_proof
        .evaluation
        .as_ref()
        .expect("generated monolithic prover emits evaluation proof");
    assert_state_history_prefix_match(
        &core_stage7.transcript_states,
        &transcript_states(monolithic_prover_transcript.log()),
    );
    assert_dory_proofs_match(
        &DoryProof(fixture.proof.joint_opening_proof.clone()),
        &monolithic_evaluation.joint_opening_proof,
    );
    assert_core_accepts_bolt_evaluation_proof(&fixture, monolithic_evaluation);
    assert_commitments_match(
        &commitment_prover_trace.commitments,
        &monolithic_artifacts.commitment.commitments,
    );
    assert_stage4_artifacts_match(&stage4_artifacts, &monolithic_artifacts.stage4);
    let monolithic_stage5_proof = to_generated_stage5_proof(&monolithic_artifacts.stage5);
    assert_stage5_artifacts_match(&generated_stage5_proof, &monolithic_stage5_proof);
    assert_stage6_artifacts_match(
        &generated_stage6_proof,
        &generated_stage6_execution_artifacts(&monolithic_artifacts.stage6),
    );
    assert_stage7_artifacts_match(
        &generated_stage7_proof,
        &generated_stage7_execution_artifacts(&monolithic_artifacts.stage7),
    );
    assert_state_history_prefix_match(
        &core_stage7.transcript_states,
        &transcript_states(monolithic_prover_transcript.log()),
    );

    let mut monolithic_verify_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut monolithic_verify_transcript, &fixture);
    let evaluation_setup = DoryScheme::verifier_setup(&fixture.pcs_setup);
    let monolithic_verified_artifacts = jolt_verifier::verify_jolt_with_programs(
        &monolithic_proof,
        jolt_verifier::JoltVerifierInputs {
            stage2_openings: &generated_jolt_stage2_openings,
            stage2_ram: Some(&generated_ram_data),
            stage3_openings: &generated_jolt_stage3_openings,
            stage4_openings: &generated_jolt_stage4_openings,
            stage5_openings: &generated_jolt_stage5_openings,
            stage6_openings: &core_stage6.opening_inputs,
            stage6_data: Some(&generated_stage6_data),
            stage7_openings: &core_stage7.opening_inputs,
            evaluation_setup: Some(&evaluation_setup),
        },
        generated_stage7_programs,
        &mut monolithic_verify_transcript,
    )
    .expect("generated monolithic verifier accepts generated monolithic prover proof");
    assert_eq!(
        monolithic_artifacts.stage6.sumchecks.len(),
        monolithic_verified_artifacts.stage6.sumchecks.len()
    );
    assert_eq!(
        monolithic_artifacts.stage7.sumchecks.len(),
        monolithic_verified_artifacts.stage7.sumchecks.len()
    );
    let core_stage8_states = core_stage8_transcript_states(&fixture);
    assert_state_history_match(
        &core_stage8_states,
        &transcript_states(monolithic_verify_transcript.log()),
    );

    let mut missing_setup_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut missing_setup_transcript, &fixture);
    let missing_setup_result = jolt_verifier::verify_jolt_with_programs(
        &monolithic_proof,
        jolt_verifier::JoltVerifierInputs {
            stage2_openings: &generated_jolt_stage2_openings,
            stage2_ram: Some(&generated_ram_data),
            stage3_openings: &generated_jolt_stage3_openings,
            stage4_openings: &generated_jolt_stage4_openings,
            stage5_openings: &generated_jolt_stage5_openings,
            stage6_openings: &core_stage6.opening_inputs,
            stage6_data: Some(&generated_stage6_data),
            stage7_openings: &core_stage7.opening_inputs,
            evaluation_setup: None,
        },
        generated_stage7_programs,
        &mut missing_setup_transcript,
    );
    assert!(
        matches!(
            missing_setup_result,
            Err(jolt_verifier::JoltVerifyError::Evaluation(
                jolt_verifier::JoltEvaluationProofError::MissingVerifierSetup
            ))
        ),
        "generated monolithic verifier accepted evaluation proof without verifier setup"
    );

    let mut missing_evaluation_proof = monolithic_proof.clone();
    missing_evaluation_proof.evaluation = None;
    let mut missing_evaluation_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut missing_evaluation_transcript, &fixture);
    let missing_evaluation_result = jolt_verifier::verify_jolt_with_programs(
        &missing_evaluation_proof,
        jolt_verifier::JoltVerifierInputs {
            stage2_openings: &generated_jolt_stage2_openings,
            stage2_ram: Some(&generated_ram_data),
            stage3_openings: &generated_jolt_stage3_openings,
            stage4_openings: &generated_jolt_stage4_openings,
            stage5_openings: &generated_jolt_stage5_openings,
            stage6_openings: &core_stage6.opening_inputs,
            stage6_data: Some(&generated_stage6_data),
            stage7_openings: &core_stage7.opening_inputs,
            evaluation_setup: Some(&evaluation_setup),
        },
        generated_stage7_programs,
        &mut missing_evaluation_transcript,
    );
    assert!(
        matches!(
            missing_evaluation_result,
            Err(jolt_verifier::JoltVerifyError::Evaluation(
                jolt_verifier::JoltEvaluationProofError::MissingProof
            ))
        ),
        "generated monolithic verifier accepted missing evaluation proof with verifier setup"
    );

    let mut tampered_evaluation_proof = monolithic_proof.clone();
    tampered_evaluation_proof
        .evaluation
        .as_mut()
        .expect("evaluation proof")
        .joint_opening_proof = unrelated_dory_proof();
    let mut tampered_evaluation_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut tampered_evaluation_transcript, &fixture);
    let tampered_evaluation_result = jolt_verifier::verify_jolt_with_programs(
        &tampered_evaluation_proof,
        jolt_verifier::JoltVerifierInputs {
            stage2_openings: &generated_jolt_stage2_openings,
            stage2_ram: Some(&generated_ram_data),
            stage3_openings: &generated_jolt_stage3_openings,
            stage4_openings: &generated_jolt_stage4_openings,
            stage5_openings: &generated_jolt_stage5_openings,
            stage6_openings: &core_stage6.opening_inputs,
            stage6_data: Some(&generated_stage6_data),
            stage7_openings: &core_stage7.opening_inputs,
            evaluation_setup: Some(&evaluation_setup),
        },
        generated_stage7_programs,
        &mut tampered_evaluation_transcript,
    );
    assert!(
        matches!(
            tampered_evaluation_result,
            Err(jolt_verifier::JoltVerifyError::Evaluation(_))
        ),
        "generated monolithic verifier accepted a tampered evaluation proof"
    );

    let assert_stage3_tamper_rejected = |tampered_stage3_artifacts: Stage3ExecutionArtifacts<
        Fr,
    >,
                                         message: &str| {
        let mut tamper_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut tamper_transcript, &fixture);
        append_bolt_commitments_to_transcript(
            &mut tamper_transcript,
            &commitment_verifier_trace.records,
            &commitment_verifier_trace.commitments,
            &commitment_verifier_program.transcript_steps,
        );
        let mut tamper_stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
        let tamper_verified_stage1 = execute_stage1_program(
            stage1_prover_plan,
            Stage1ExecutionMode::Verifier,
            &mut tamper_stage1_verifier,
            &mut tamper_transcript,
        )
        .expect("Bolt Stage 1 verifier accepts before Stage 3 tamper");
        let tamper_stage2_openings = stage2_opening_inputs(&tamper_verified_stage1);
        let mut tamper_stage2_verifier =
            Stage2VerifierKernelExecutor::new(&stage2_proof, &tamper_stage2_openings)
                .with_ram_data(&ram_data);
        let tamper_verified_stage2 = execute_stage2_program(
            stage2_prover_plan,
            Stage2ExecutionMode::Verifier,
            &mut tamper_stage2_verifier,
            &mut tamper_transcript,
        )
        .expect("Bolt Stage 2 verifier accepts before Stage 3 tamper");
        let tamper_stage3_openings =
            stage3_opening_inputs(&tamper_verified_stage1, &tamper_verified_stage2);
        let tampered_stage3_proof = Stage3Proof::from(tampered_stage3_artifacts.clone());
        let mut tamper_stage3_verifier =
            Stage3VerifierKernelExecutor::new(&tampered_stage3_proof, &tamper_stage3_openings);
        let tamper_result = execute_stage3_program(
            stage3_prover_plan,
            Stage3ExecutionMode::Verifier,
            &mut tamper_stage3_verifier,
            &mut tamper_transcript,
        );
        assert!(tamper_result.is_err(), "{message}");

        let mut generated_tamper_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut generated_tamper_transcript, &fixture);
        append_bolt_commitments_to_transcript(
            &mut generated_tamper_transcript,
            &commitment_verifier_trace.records,
            &commitment_verifier_trace.commitments,
            &commitment_verifier_program.transcript_steps,
        );
        let mut generated_tamper_stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
        let generated_tamper_verified_stage1 = execute_stage1_program(
            stage1_prover_plan,
            Stage1ExecutionMode::Verifier,
            &mut generated_tamper_stage1_verifier,
            &mut generated_tamper_transcript,
        )
        .expect("Bolt Stage 1 verifier accepts before generated Stage 3 tamper");
        let generated_tamper_stage2_openings =
            stage2_opening_inputs(&generated_tamper_verified_stage1);
        let mut generated_tamper_stage2_verifier =
            Stage2VerifierKernelExecutor::new(&stage2_proof, &generated_tamper_stage2_openings)
                .with_ram_data(&ram_data);
        let generated_tamper_verified_stage2 = execute_stage2_program(
            stage2_prover_plan,
            Stage2ExecutionMode::Verifier,
            &mut generated_tamper_stage2_verifier,
            &mut generated_tamper_transcript,
        )
        .expect("Bolt Stage 2 verifier accepts before generated Stage 3 tamper");
        let generated_tamper_stage3_openings =
            generated_stage3_opening_inputs(&stage3_opening_inputs(
                &generated_tamper_verified_stage1,
                &generated_tamper_verified_stage2,
            ));
        let generated_tampered_stage3_proof = to_generated_stage3_proof(&tampered_stage3_artifacts);
        let generated_tamper_result = generated_stage3::verify_stage3_with_program(
            generated_stage3_verifier_plan,
            &generated_tampered_stage3_proof,
            &generated_tamper_stage3_openings,
            &mut generated_tamper_transcript,
        );
        assert!(generated_tamper_result.is_err(), "generated {message}");

        let generated_tampered_jolt_proof = to_generated_jolt_proof(
            &commitment_verifier_trace.commitments,
            &stage1_artifacts,
            &stage2_artifacts,
            &tampered_stage3_artifacts,
            &stage4_artifacts,
            &generated_stage5_proof,
        );
        let mut generated_jolt_tamper_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut generated_jolt_tamper_transcript, &fixture);
        let generated_jolt_tamper_result = jolt_verifier::verify_jolt_with_programs(
            &generated_tampered_jolt_proof,
            jolt_verifier::JoltVerifierInputs {
                stage2_openings: &generated_jolt_stage2_openings,
                stage2_ram: Some(&generated_ram_data),
                stage3_openings: &generated_jolt_stage3_openings,
                stage4_openings: &generated_jolt_stage4_openings,
                stage5_openings: &generated_jolt_stage5_openings,
                stage6_openings: &[],
                stage6_data: Some(&generated_stage6_data),
                stage7_openings: &[],
                evaluation_setup: None,
            },
            generated_programs,
            &mut generated_jolt_tamper_transcript,
        );
        assert!(
            generated_jolt_tamper_result.is_err(),
            "generated monolithic {message}"
        );
    };

    let mut tampered_sumcheck = stage3_artifacts.clone();
    let tampered_poly = &mut tampered_sumcheck.sumchecks[0].proof.round_polynomials[0];
    let mut tampered_coefficients = tampered_poly.clone().into_coefficients();
    tampered_coefficients[0] += Fr::from_u64(1);
    *tampered_poly = UnivariatePoly::new(tampered_coefficients);
    assert_stage3_tamper_rejected(
        tampered_sumcheck,
        "Bolt Stage 3 verifier accepted a tampered batched sumcheck coefficient",
    );

    let mut tampered_eval = stage3_artifacts.clone();
    tampered_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    assert_stage3_tamper_rejected(
        tampered_eval,
        "Bolt Stage 3 verifier accepted a tampered batched opening evaluation",
    );

    let mut tampered_point = stage3_artifacts.clone();
    tampered_point.sumchecks[0].point[0] += Fr::from_u64(1);
    assert_stage3_tamper_rejected(
        tampered_point,
        "Bolt Stage 3 verifier accepted a tampered batched sumcheck point",
    );

    let assert_stage4_tamper_rejected = |tampered_stage4_artifacts: Stage4ExecutionArtifacts<
        Fr,
    >,
                                         message: &str| {
        let mut generated_tamper_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut generated_tamper_transcript, &fixture);
        append_bolt_commitments_to_transcript(
            &mut generated_tamper_transcript,
            &commitment_verifier_trace.records,
            &commitment_verifier_trace.commitments,
            &commitment_verifier_program.transcript_steps,
        );
        let mut generated_tamper_stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
        let generated_tamper_verified_stage1 = execute_stage1_program(
            stage1_prover_plan,
            Stage1ExecutionMode::Verifier,
            &mut generated_tamper_stage1_verifier,
            &mut generated_tamper_transcript,
        )
        .expect("Bolt Stage 1 verifier accepts before generated Stage 4 tamper");
        let generated_tamper_stage2_openings =
            stage2_opening_inputs(&generated_tamper_verified_stage1);
        let mut generated_tamper_stage2_verifier =
            Stage2VerifierKernelExecutor::new(&stage2_proof, &generated_tamper_stage2_openings)
                .with_ram_data(&ram_data);
        let generated_tamper_verified_stage2 = execute_stage2_program(
            stage2_prover_plan,
            Stage2ExecutionMode::Verifier,
            &mut generated_tamper_stage2_verifier,
            &mut generated_tamper_transcript,
        )
        .expect("Bolt Stage 2 verifier accepts before generated Stage 4 tamper");
        let generated_tamper_stage3_openings =
            generated_stage3_opening_inputs(&stage3_opening_inputs(
                &generated_tamper_verified_stage1,
                &generated_tamper_verified_stage2,
            ));
        let generated_tamper_stage3_proof = to_generated_stage3_proof(&stage3_artifacts);
        let _ = generated_stage3::verify_stage3_with_program(
            generated_stage3_verifier_plan,
            &generated_tamper_stage3_proof,
            &generated_tamper_stage3_openings,
            &mut generated_tamper_transcript,
        )
        .expect("generated Stage 3 verifier accepts before generated Stage 4 tamper");
        let generated_tampered_stage4_proof = to_generated_stage4_proof(&tampered_stage4_artifacts);
        let generated_tamper_result = generated_stage4::verify_stage4_with_program(
            generated_stage4_verifier_plan,
            &generated_tampered_stage4_proof,
            &generated_stage4_openings,
            &mut generated_tamper_transcript,
        );
        assert!(generated_tamper_result.is_err(), "generated {message}");

        let generated_tampered_jolt_proof = to_generated_jolt_proof(
            &commitment_verifier_trace.commitments,
            &stage1_artifacts,
            &stage2_artifacts,
            &stage3_artifacts,
            &tampered_stage4_artifacts,
            &generated_stage5_proof,
        );
        let mut generated_jolt_tamper_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut generated_jolt_tamper_transcript, &fixture);
        let generated_jolt_tamper_result = jolt_verifier::verify_jolt_with_programs(
            &generated_tampered_jolt_proof,
            jolt_verifier::JoltVerifierInputs {
                stage2_openings: &generated_jolt_stage2_openings,
                stage2_ram: Some(&generated_ram_data),
                stage3_openings: &generated_jolt_stage3_openings,
                stage4_openings: &generated_jolt_stage4_openings,
                stage5_openings: &generated_jolt_stage5_openings,
                stage6_openings: &[],
                stage6_data: Some(&generated_stage6_data),
                stage7_openings: &[],
                evaluation_setup: None,
            },
            generated_programs,
            &mut generated_jolt_tamper_transcript,
        );
        assert!(
            generated_jolt_tamper_result.is_err(),
            "generated monolithic {message}"
        );
    };

    let mut tampered_stage4_sumcheck = stage4_artifacts.clone();
    let tampered_stage4_poly = &mut tampered_stage4_sumcheck.sumchecks[0]
        .proof
        .round_polynomials[0];
    let mut tampered_stage4_coefficients = tampered_stage4_poly.clone().into_coefficients();
    tampered_stage4_coefficients[0] += Fr::from_u64(1);
    *tampered_stage4_poly = UnivariatePoly::new(tampered_stage4_coefficients);
    assert_stage4_tamper_rejected(
        tampered_stage4_sumcheck,
        "Stage 4 verifier accepted a tampered batched sumcheck coefficient",
    );

    let mut tampered_stage4_eval = stage4_artifacts.clone();
    tampered_stage4_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    assert_stage4_tamper_rejected(
        tampered_stage4_eval,
        "Stage 4 verifier accepted a tampered batched opening evaluation",
    );

    let mut tampered_stage4_point = stage4_artifacts.clone();
    tampered_stage4_point.sumchecks[0].point[0] += Fr::from_u64(1);
    assert_stage4_tamper_rejected(
        tampered_stage4_point,
        "Stage 4 verifier accepted a tampered batched sumcheck point",
    );

    let assert_stage5_tamper_rejected =
        |tampered_stage5_proof: generated_stage5::Stage5Proof<Fr>, message: &str| {
            let mut generated_tamper_transcript = generated_stage5_start_transcript.clone();
            let generated_tamper_result = generated_stage5::verify_stage5_with_program(
                generated_stage5_verifier_plan,
                &tampered_stage5_proof,
                &generated_stage5_openings,
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");

            let generated_tampered_jolt_proof = to_generated_jolt_proof(
                &commitment_verifier_trace.commitments,
                &stage1_artifacts,
                &stage2_artifacts,
                &stage3_artifacts,
                &stage4_artifacts,
                &tampered_stage5_proof,
            );
            let mut generated_jolt_tamper_transcript = CheckpointTranscript::<
                jolt_transcript::Blake2bTranscript<Fr>,
            >::new(TRANSCRIPT_LABEL);
            append_bolt_preamble(&mut generated_jolt_tamper_transcript, &fixture);
            let generated_jolt_tamper_result = jolt_verifier::verify_jolt_with_programs(
                &generated_tampered_jolt_proof,
                jolt_verifier::JoltVerifierInputs {
                    stage2_openings: &generated_jolt_stage2_openings,
                    stage2_ram: Some(&generated_ram_data),
                    stage3_openings: &generated_jolt_stage3_openings,
                    stage4_openings: &generated_jolt_stage4_openings,
                    stage5_openings: &generated_jolt_stage5_openings,
                    stage6_openings: &[],
                    stage6_data: Some(&generated_stage6_data),
                    stage7_openings: &[],
                    evaluation_setup: None,
                },
                generated_programs,
                &mut generated_jolt_tamper_transcript,
            );
            assert!(
                generated_jolt_tamper_result.is_err(),
                "generated monolithic {message}"
            );
        };

    let mut tampered_stage5_sumcheck = generated_stage5_proof.clone();
    let tampered_stage5_poly = &mut tampered_stage5_sumcheck.sumchecks[0]
        .proof
        .round_polynomials[0];
    let mut tampered_stage5_coefficients = tampered_stage5_poly.clone().into_coefficients();
    tampered_stage5_coefficients[0] += Fr::from_u64(1);
    *tampered_stage5_poly = UnivariatePoly::new(tampered_stage5_coefficients);
    assert_stage5_tamper_rejected(
        tampered_stage5_sumcheck,
        "Stage 5 verifier accepted a tampered batched sumcheck coefficient",
    );

    let mut tampered_stage5_eval = generated_stage5_proof.clone();
    tampered_stage5_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    assert_stage5_tamper_rejected(
        tampered_stage5_eval,
        "Stage 5 verifier accepted a tampered batched opening evaluation",
    );

    let mut tampered_stage5_point = generated_stage5_proof.clone();
    tampered_stage5_point.sumchecks[0].point[0] += Fr::from_u64(1);
    assert_stage5_tamper_rejected(
        tampered_stage5_point,
        "Stage 5 verifier accepted a tampered batched sumcheck point",
    );

    let assert_stage6_tamper_rejected =
        |tampered_stage6_proof: generated_stage6::Stage6Proof<Fr>, message: &str| {
            let mut generated_tamper_transcript = generated_verifier_transcript.clone();
            let generated_tamper_result = generated_stage6::verify_stage6_with_program(
                generated_stage6_verifier_plan,
                &tampered_stage6_proof,
                &core_stage6.opening_inputs,
                Some(&generated_stage6_data),
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");

            let generated_tampered_jolt_proof = to_generated_jolt_proof_with_stage6(
                &commitment_verifier_trace.commitments,
                &stage1_artifacts,
                &stage2_artifacts,
                &stage3_artifacts,
                &stage4_artifacts,
                &generated_stage5_proof,
                &tampered_stage6_proof,
            );
            let mut generated_jolt_tamper_transcript = CheckpointTranscript::<
                jolt_transcript::Blake2bTranscript<Fr>,
            >::new(TRANSCRIPT_LABEL);
            append_bolt_preamble(&mut generated_jolt_tamper_transcript, &fixture);
            let generated_jolt_tamper_result = jolt_verifier::verify_jolt_with_programs(
                &generated_tampered_jolt_proof,
                jolt_verifier::JoltVerifierInputs {
                    stage2_openings: &generated_jolt_stage2_openings,
                    stage2_ram: Some(&generated_ram_data),
                    stage3_openings: &generated_jolt_stage3_openings,
                    stage4_openings: &generated_jolt_stage4_openings,
                    stage5_openings: &generated_jolt_stage5_openings,
                    stage6_openings: &core_stage6.opening_inputs,
                    stage6_data: Some(&generated_stage6_data),
                    stage7_openings: &[],
                    evaluation_setup: None,
                },
                generated_stage6_programs,
                &mut generated_jolt_tamper_transcript,
            );
            assert!(
                generated_jolt_tamper_result.is_err(),
                "generated monolithic {message}"
            );
        };

    let mut tampered_stage6_sumcheck = generated_stage6_proof.clone();
    let tampered_stage6_poly = &mut tampered_stage6_sumcheck.sumchecks[0]
        .proof
        .round_polynomials[0];
    let mut tampered_stage6_coefficients = tampered_stage6_poly.clone().into_coefficients();
    tampered_stage6_coefficients[0] += Fr::from_u64(1);
    *tampered_stage6_poly = UnivariatePoly::new(tampered_stage6_coefficients);
    assert_stage6_tamper_rejected(
        tampered_stage6_sumcheck,
        "Stage 6 verifier accepted a tampered batched sumcheck coefficient",
    );

    let mut tampered_stage6_eval = generated_stage6_proof.clone();
    tampered_stage6_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    assert_stage6_tamper_rejected(
        tampered_stage6_eval,
        "Stage 6 verifier accepted a tampered batched opening evaluation",
    );

    let mut tampered_stage6_point = generated_stage6_proof.clone();
    tampered_stage6_point.sumchecks[0].point[0] += Fr::from_u64(1);
    assert_stage6_tamper_rejected(
        tampered_stage6_point,
        "Stage 6 verifier accepted a tampered batched sumcheck point",
    );

    let assert_stage7_tamper_rejected =
        |tampered_stage7_proof: generated_stage7::Stage7Proof<Fr>, message: &str| {
            let mut generated_tamper_transcript = generated_stage6_transcript.clone();
            let generated_tamper_result = generated_stage7::verify_stage7_with_program(
                generated_stage7_verifier_plan,
                &tampered_stage7_proof,
                &core_stage7.opening_inputs,
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");

            let generated_tampered_jolt_proof = to_generated_jolt_proof_with_stage7(
                &commitment_verifier_trace.commitments,
                &stage1_artifacts,
                &stage2_artifacts,
                &stage3_artifacts,
                &stage4_artifacts,
                &generated_stage5_proof,
                &generated_stage6_proof,
                &tampered_stage7_proof,
            );
            let mut generated_jolt_tamper_transcript = CheckpointTranscript::<
                jolt_transcript::Blake2bTranscript<Fr>,
            >::new(TRANSCRIPT_LABEL);
            append_bolt_preamble(&mut generated_jolt_tamper_transcript, &fixture);
            let generated_jolt_tamper_result = jolt_verifier::verify_jolt_with_programs(
                &generated_tampered_jolt_proof,
                jolt_verifier::JoltVerifierInputs {
                    stage2_openings: &generated_jolt_stage2_openings,
                    stage2_ram: Some(&generated_ram_data),
                    stage3_openings: &generated_jolt_stage3_openings,
                    stage4_openings: &generated_jolt_stage4_openings,
                    stage5_openings: &generated_jolt_stage5_openings,
                    stage6_openings: &core_stage6.opening_inputs,
                    stage6_data: Some(&generated_stage6_data),
                    stage7_openings: &core_stage7.opening_inputs,
                    evaluation_setup: None,
                },
                generated_stage7_programs,
                &mut generated_jolt_tamper_transcript,
            );
            assert!(
                generated_jolt_tamper_result.is_err(),
                "generated monolithic {message}"
            );
        };

    let mut tampered_stage7_sumcheck = generated_stage7_proof.clone();
    let tampered_stage7_poly = &mut tampered_stage7_sumcheck.sumchecks[0]
        .proof
        .round_polynomials[0];
    let mut tampered_stage7_coefficients = tampered_stage7_poly.clone().into_coefficients();
    tampered_stage7_coefficients[0] += Fr::from_u64(1);
    *tampered_stage7_poly = UnivariatePoly::new(tampered_stage7_coefficients);
    assert_stage7_tamper_rejected(
        tampered_stage7_sumcheck,
        "Stage 7 verifier accepted a tampered batched sumcheck coefficient",
    );

    let mut tampered_stage7_eval = generated_stage7_proof.clone();
    tampered_stage7_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    assert_stage7_tamper_rejected(
        tampered_stage7_eval,
        "Stage 7 verifier accepted a tampered batched opening evaluation",
    );

    let mut tampered_stage7_point = generated_stage7_proof.clone();
    tampered_stage7_point.sumchecks[0].point[0] += Fr::from_u64(1);
    assert_stage7_tamper_rejected(
        tampered_stage7_point,
        "Stage 7 verifier accepted a tampered batched sumcheck point",
    );
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

struct GeneratedCommitmentOracleMap {
    data: BTreeMap<String, Option<Vec<Fr>>>,
}

impl generated_prover_commitment::CommitmentInputProvider for GeneratedCommitmentOracleMap {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {
        self.data
            .get(oracle)
            .and_then(|values| values.as_ref())
            .map(|values| Cow::Borrowed(values.as_slice()))
    }
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
    rv64_cycles: Vec<Stage1Rv64Cycle>,
    core_rv64_cycles: Vec<Stage1Rv64Cycle>,
    product_virtual_cycles: Vec<Stage2ProductVirtualCycle>,
    instruction_lookup_cycles: Vec<Stage2InstructionLookupCycle>,
    stage3_cycles: Vec<Stage3Cycle>,
    stage4_register_accesses: Vec<Stage4RegisterAccess>,
    stage5_lookup_indices: Vec<u128>,
    stage5_lookup_table_indices: Vec<Option<usize>>,
    stage5_is_interleaved_operands: Vec<bool>,
    padded_trace: Vec<jolt_trace::Cycle>,
    stage6_bytecode_entries: Vec<generated_stage6::Stage6BytecodeEntry>,
    stage6_entry_bytecode_index: usize,
    ram_accesses: Vec<Stage2RamAccess>,
    initial_ram_state: Vec<u64>,
    final_ram_state: Vec<u64>,
    ram_start_address: u64,
    ram_output_layout: Stage2RamOutputLayout,
    commitments: Vec<CoreCommitment>,
}

#[derive(Clone, Debug)]
struct CoreStage4Artifacts {
    artifacts: Stage4ExecutionArtifacts<Fr>,
    opening_inputs: Vec<Stage4OpeningInputValue<Fr>>,
    transcript_states: Vec<[u8; 32]>,
}

#[derive(Clone, Debug)]
struct CoreStage5Artifacts {
    proof: generated_stage5::Stage5Proof<Fr>,
    opening_inputs: Vec<generated_stage5::Stage5OpeningInputValue<Fr>>,
    transcript_states: Vec<[u8; 32]>,
}

#[derive(Clone, Debug)]
struct CoreStage6Artifacts {
    proof: generated_stage6::Stage6Proof<Fr>,
    opening_inputs: Vec<generated_stage6::Stage6OpeningInputValue<Fr>>,
    transcript_states: Vec<[u8; 32]>,
}

#[derive(Clone, Debug)]
struct CoreStage7Artifacts {
    proof: generated_stage7::Stage7Proof<Fr>,
    opening_inputs: Vec<generated_stage7::Stage7OpeningInputValue<Fr>>,
    transcript_states: Vec<[u8; 32]>,
}

#[derive(Clone, Debug)]
struct Stage6WitnessPolynomials {
    instruction_ra_booleanity: Vec<Vec<Fr>>,
    bytecode_ra_booleanity: Vec<Vec<Fr>>,
    ram_ra_booleanity: Vec<Vec<Fr>>,
    bytecode_ra_read_raf: Vec<Vec<Fr>>,
    instruction_ra_virtual: Vec<Vec<Fr>>,
    ram_ra_virtual: Vec<Vec<Fr>>,
    hamming_weight: Vec<Fr>,
    ram_inc: Vec<Fr>,
    rd_inc: Vec<Fr>,
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

fn bolt_stage3_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage3CpuProgram, CompilerStage3CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage3_protocol(&context, params).expect("build stage3 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 3 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage3_to_compute(&context, &prover_party).expect("lower prover Stage 3");
    let verifier_compute =
        lower_stage3_to_compute(&context, &verifier_party).expect("lower verifier Stage 3");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage3_cpu_program(&prover_cpu).expect("extract prover Stage 3 CPU");
    let verifier_program = stage3_cpu_program(&verifier_cpu).expect("extract verifier Stage 3 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage4_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage4CpuProgram, CompilerStage4CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage4_protocol(&context, params).expect("build stage4 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 4 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage4_to_compute(&context, &prover_party).expect("lower prover Stage 4");
    let verifier_compute =
        lower_stage4_to_compute(&context, &verifier_party).expect("lower verifier Stage 4");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage4_cpu_program(&prover_cpu).expect("extract prover Stage 4 CPU");
    let verifier_program = stage4_cpu_program(&verifier_cpu).expect("extract verifier Stage 4 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage5_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage5CpuProgram, CompilerStage5CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage5_protocol(&context, params).expect("build stage5 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 5 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage5_to_compute(&context, &prover_party).expect("lower prover Stage 5");
    let verifier_compute =
        lower_stage5_to_compute(&context, &verifier_party).expect("lower verifier Stage 5");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage5_cpu_program(&prover_cpu).expect("extract prover Stage 5 CPU");
    let verifier_program = stage5_cpu_program(&verifier_cpu).expect("extract verifier Stage 5 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage6_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage6CpuProgram, CompilerStage6CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage6_protocol(&context, params).expect("build stage6 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 6 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage6_to_compute(&context, &prover_party).expect("lower prover Stage 6");
    let verifier_compute =
        lower_stage6_to_compute(&context, &verifier_party).expect("lower verifier Stage 6");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage6_cpu_program(&prover_cpu).expect("extract prover Stage 6 CPU");
    let verifier_program = stage6_cpu_program(&verifier_cpu).expect("extract verifier Stage 6 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage7_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage7CpuProgram, CompilerStage7CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage7_protocol(&context, params).expect("build stage7 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 7 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage7_to_compute(&context, &prover_party).expect("lower prover Stage 7");
    let verifier_compute =
        lower_stage7_to_compute(&context, &verifier_party).expect("lower verifier Stage 7");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage7_cpu_program(&prover_cpu).expect("extract prover Stage 7 CPU");
    let verifier_program = stage7_cpu_program(&verifier_cpu).expect("extract verifier Stage 7 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage8_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage8CpuProgram, CompilerStage8CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage8_protocol(&context, params).expect("build stage8 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 8 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage8_to_compute(&context, &prover_party).expect("lower prover Stage 8");
    let verifier_compute =
        lower_stage8_to_compute(&context, &verifier_party).expect("lower verifier Stage 8");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage8_cpu_program(&prover_cpu).expect("extract prover Stage 8 CPU");
    let verifier_program = stage8_cpu_program(&verifier_cpu).expect("extract verifier Stage 8 CPU");
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
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .filter(|plan| plan.symbol.starts_with("stage2.product_virtual.uniskip."))
                .map(|plan| KernelStage2FieldExprPlan {
                    symbol: leak_str(&plan.symbol),
                    kind: leak_str(&plan.kind),
                    formula: leak_str(&plan.formula),
                    operand_names: leak_str_slice(&plan.operand_names),
                    operands: leak_str_slice(&plan.operands),
                })
                .collect(),
        ),
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

fn leak_stage3_program(program: &CompilerStage3CpuProgram) -> &'static KernelStage3CpuProgramPlan {
    Box::leak(Box::new(KernelStage3CpuProgramPlan {
        params: Stage3Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| KernelStage3ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| KernelStage3TranscriptSqueezePlan {
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
                .map(|plan| Stage3OpeningInputPlan {
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
                .map(|plan| KernelStage3FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| KernelStage3FieldExprPlan {
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
                .map(|plan| KernelStage3KernelPlan {
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
                .map(|plan| KernelStage3SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage3 claim kernel")),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| KernelStage3SumcheckBatchPlan {
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
                .map(|plan| KernelStage3SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage3 driver kernel")),
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
                .map(|plan| Stage3SumcheckInstanceResultPlan {
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
                .map(|plan| KernelStage3SumcheckEvalPlan {
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
                .map(|plan| Stage3PointSlicePlan {
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
                .map(|plan| Stage3PointConcatPlan {
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
                .map(|plan| KernelStage3OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| KernelStage3OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| KernelStage3OpeningBatchPlan {
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

#[allow(dead_code)]
fn leak_stage4_program(program: &CompilerStage4CpuProgram) -> &'static KernelStage4CpuProgramPlan {
    Box::leak(Box::new(KernelStage4CpuProgramPlan {
        role: role_name(&program.role),
        params: Stage4Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| KernelStage4ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| KernelStage4TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| KernelStage4TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| Stage4OpeningInputPlan {
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
                .map(|plan| KernelStage4FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| KernelStage4FieldExprPlan {
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
                .map(|plan| KernelStage4KernelPlan {
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
                .map(|plan| KernelStage4SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| KernelStage4SumcheckBatchPlan {
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
                .map(|plan| KernelStage4SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| Stage4SumcheckInstanceResultPlan {
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
                .map(|plan| KernelStage4SumcheckEvalPlan {
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
                .map(|plan| Stage4PointSlicePlan {
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
                .map(|plan| Stage4PointConcatPlan {
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
                .map(|plan| KernelStage4OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| KernelStage4OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| KernelStage4OpeningBatchPlan {
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

#[allow(dead_code)]
fn leak_stage5_program(
    program: &CompilerStage5CpuProgram,
) -> &'static kernel_stage5::Stage5CpuProgramPlan {
    Box::leak(Box::new(kernel_stage5::Stage5CpuProgramPlan {
        role: role_name(&program.role),
        params: kernel_stage5::Stage5Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| kernel_stage5::Stage5ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| kernel_stage5::Stage5TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| kernel_stage5::Stage5TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| kernel_stage5::Stage5OpeningInputPlan {
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
                .map(|plan| kernel_stage5::Stage5FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| kernel_stage5::Stage5FieldExprPlan {
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
                .map(|plan| kernel_stage5::Stage5KernelPlan {
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
                .map(|plan| kernel_stage5::Stage5SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| kernel_stage5::Stage5SumcheckBatchPlan {
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
                .map(|plan| kernel_stage5::Stage5SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| kernel_stage5::Stage5SumcheckInstanceResultPlan {
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
                .map(|plan| kernel_stage5::Stage5SumcheckEvalPlan {
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
                .map(|plan| kernel_stage5::Stage5PointSlicePlan {
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
                .map(|plan| kernel_stage5::Stage5PointConcatPlan {
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
                .map(|plan| kernel_stage5::Stage5OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| kernel_stage5::Stage5OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| kernel_stage5::Stage5OpeningBatchPlan {
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

#[allow(dead_code)]
fn leak_stage6_program(
    program: &CompilerStage6CpuProgram,
) -> &'static kernel_stage6::Stage6CpuProgramPlan {
    Box::leak(Box::new(kernel_stage6::Stage6CpuProgramPlan {
        role: role_name(&program.role),
        params: kernel_stage6::Stage6Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| kernel_stage6::Stage6ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| kernel_stage6::Stage6TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| kernel_stage6::Stage6TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| kernel_stage6::Stage6OpeningInputPlan {
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
                .map(|plan| kernel_stage6::Stage6FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| kernel_stage6::Stage6FieldExprPlan {
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
                .map(|plan| kernel_stage6::Stage6KernelPlan {
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
                .map(|plan| kernel_stage6::Stage6SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| kernel_stage6::Stage6SumcheckBatchPlan {
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
                .map(|plan| kernel_stage6::Stage6SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| kernel_stage6::Stage6SumcheckInstanceResultPlan {
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
                .map(|plan| kernel_stage6::Stage6SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_zeros: leak_slice(
            program
                .point_zeros
                .iter()
                .map(|plan| kernel_stage6::Stage6PointZeroPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    arity: plan.arity,
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| kernel_stage6::Stage6PointSlicePlan {
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
                .map(|plan| kernel_stage6::Stage6PointConcatPlan {
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
                .map(|plan| kernel_stage6::Stage6OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| kernel_stage6::Stage6OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| kernel_stage6::Stage6OpeningBatchPlan {
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

fn leak_stage7_program(
    program: &CompilerStage7CpuProgram,
) -> &'static jolt_kernels::stage7::Stage7CpuProgramPlan {
    Box::leak(Box::new(jolt_kernels::stage7::Stage7CpuProgramPlan {
        role: role_name(&program.role),
        params: jolt_kernels::stage7::Stage7Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(
                    |plan| jolt_kernels::stage7::Stage7TranscriptAbsorbBytesPlan {
                        symbol: leak_str(&plan.symbol),
                        label: leak_str(&plan.label),
                        payload: leak_str(&plan.payload),
                    },
                )
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7OpeningInputPlan {
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
                .map(|plan| jolt_kernels::stage7::Stage7FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7FieldExprPlan {
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
                .map(|plan| jolt_kernels::stage7::Stage7KernelPlan {
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
                .map(|plan| jolt_kernels::stage7::Stage7SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7SumcheckBatchPlan {
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
                .map(|plan| jolt_kernels::stage7::Stage7SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(
                    |plan| jolt_kernels::stage7::Stage7SumcheckInstanceResultPlan {
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
                    },
                )
                .collect(),
        ),
        evals: leak_slice(
            program
                .evals
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_zeros: leak_slice(
            program
                .point_zeros
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7PointZeroPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    arity: plan.arity,
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7PointSlicePlan {
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
                .map(|plan| jolt_kernels::stage7::Stage7PointConcatPlan {
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
                .map(|plan| jolt_kernels::stage7::Stage7OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(
                    |plan| jolt_kernels::stage7::Stage7OpeningClaimEqualityPlan {
                        symbol: leak_str(&plan.symbol),
                        mode: leak_str(&plan.mode),
                        lhs: leak_str(&plan.lhs),
                        rhs: leak_str(&plan.rhs),
                    },
                )
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| jolt_kernels::stage7::Stage7OpeningBatchPlan {
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

fn leak_generated_commitment_prover_program(
    program: &CommitmentCpuProgram,
) -> &'static generated_prover_commitment::CommitmentProverProgramPlan {
    Box::leak(Box::new(
        generated_prover_commitment::CommitmentProverProgramPlan {
            params: generated_prover_commitment::CommitmentParams {
                field: leak_str(&program.params.field),
                pcs: leak_str(&program.params.pcs),
                transcript: leak_str(&program.params.transcript),
            },
            oracle_plans: leak_slice(
                program
                    .oracle_plans
                    .iter()
                    .map(|plan| generated_prover_commitment::OraclePlan {
                        oracle: leak_str(&plan.oracle),
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                    })
                    .collect(),
            ),
            batch_plans: leak_slice(
                program
                    .batch_plans
                    .iter()
                    .map(|plan| generated_prover_commitment::CommitmentBatchPlan {
                        artifact: leak_str(&plan.artifact),
                        pcs: leak_str(&plan.pcs),
                        oracle_family: leak_str(&plan.oracle_family),
                        label: leak_str(&plan.label),
                        oracles: leak_str_slice(&plan.oracles),
                        count: plan.count,
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                    })
                    .collect(),
            ),
            optional_plans: leak_slice(
                program
                    .optional_plans
                    .iter()
                    .map(|plan| generated_prover_commitment::OptionalCommitmentPlan {
                        artifact: leak_str(&plan.artifact),
                        pcs: leak_str(&plan.pcs),
                        oracle: leak_str(&plan.oracle),
                        label: leak_str(&plan.label),
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                        skip_policy: match plan.skip_policy {
                            OptionalSkipPolicy::MissingOrZero => {
                                generated_prover_commitment::OptionalSkipPolicy::MissingOrZero
                            }
                        },
                    })
                    .collect(),
            ),
            transcript_steps: leak_slice(
                program
                    .transcript_steps
                    .iter()
                    .map(|step| generated_prover_commitment::TranscriptStep {
                        label: leak_str(&step.label),
                        source: leak_str(&step.source),
                        optional: step.optional,
                    })
                    .collect(),
            ),
        },
    ))
}

fn leak_generated_commitment_verifier_program(
    program: &CommitmentCpuProgram,
) -> &'static generated_commitment::CommitmentVerifierProgramPlan {
    Box::leak(Box::new(
        generated_commitment::CommitmentVerifierProgramPlan {
            params: generated_commitment::CommitmentParams {
                field: leak_str(&program.params.field),
                pcs: leak_str(&program.params.pcs),
                transcript: leak_str(&program.params.transcript),
            },
            oracle_plans: leak_slice(
                program
                    .oracle_plans
                    .iter()
                    .map(|plan| generated_commitment::OraclePlan {
                        oracle: leak_str(&plan.oracle),
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                    })
                    .collect(),
            ),
            batch_plans: leak_slice(
                program
                    .batch_plans
                    .iter()
                    .map(|plan| generated_commitment::CommitmentBatchPlan {
                        artifact: leak_str(&plan.artifact),
                        pcs: leak_str(&plan.pcs),
                        oracle_family: leak_str(&plan.oracle_family),
                        label: leak_str(&plan.label),
                        oracles: leak_str_slice(&plan.oracles),
                        count: plan.count,
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                    })
                    .collect(),
            ),
            optional_plans: leak_slice(
                program
                    .optional_plans
                    .iter()
                    .map(|plan| generated_commitment::OptionalCommitmentPlan {
                        artifact: leak_str(&plan.artifact),
                        pcs: leak_str(&plan.pcs),
                        oracle: leak_str(&plan.oracle),
                        label: leak_str(&plan.label),
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                        skip_policy: match plan.skip_policy {
                            OptionalSkipPolicy::MissingOrZero => {
                                generated_commitment::OptionalSkipPolicy::MissingOrZero
                            }
                        },
                    })
                    .collect(),
            ),
            transcript_steps: leak_slice(
                program
                    .transcript_steps
                    .iter()
                    .map(|step| generated_commitment::TranscriptStep {
                        label: leak_str(&step.label),
                        source: leak_str(&step.source),
                        optional: step.optional,
                    })
                    .collect(),
            ),
        },
    ))
}

fn leak_generated_stage1_verifier_program(
    program: &CompilerStage1CpuProgram,
) -> &'static generated_stage1::Stage1VerifierProgramPlan {
    Box::leak(Box::new(generated_stage1::Stage1VerifierProgramPlan {
        params: generated_stage1::Stage1Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| generated_stage1::Stage1TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        claims: leak_slice(
            program
                .claims
                .iter()
                .map(|plan| generated_stage1::Stage1SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    relation: leak_str(
                        plan.relation
                            .as_deref()
                            .expect("Stage1 verifier claim relation"),
                    ),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| generated_stage1::Stage1SumcheckBatchPlan {
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
                .map(|plan| generated_stage1::Stage1SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    relation: leak_str(
                        plan.relation
                            .as_deref()
                            .expect("Stage1 verifier driver relation"),
                    ),
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
                .map(|plan| generated_stage1::Stage1SumcheckInstanceResultPlan {
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
                .map(|plan| generated_stage1::Stage1SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        opening_claims: leak_slice(
            program
                .opening_claims
                .iter()
                .map(|plan| generated_stage1::Stage1OpeningClaimPlan {
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
                .map(|plan| generated_stage1::Stage1OpeningBatchPlan {
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

fn leak_generated_stage2_verifier_program(
    program: &CompilerStage2CpuProgram,
) -> &'static generated_stage2::Stage2VerifierProgramPlan {
    Box::leak(Box::new(generated_stage2::Stage2VerifierProgramPlan {
        params: generated_stage2::Stage2Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| generated_stage2::Stage2ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| generated_stage2::Stage2TranscriptSqueezePlan {
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
                .map(|plan| generated_stage2::Stage2OpeningInputPlan {
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
                .map(|plan| generated_stage2::Stage2FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| generated_stage2::Stage2FieldExprPlan {
                    symbol: leak_str(&plan.symbol),
                    kind: leak_str(&plan.kind),
                    formula: leak_str(&plan.formula),
                    operand_names: leak_str_slice(&plan.operand_names),
                    operands: leak_str_slice(&plan.operands),
                })
                .collect(),
        ),
        claims: leak_slice(
            program
                .claims
                .iter()
                .map(|plan| generated_stage2::Stage2SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    relation: leak_str(
                        plan.relation
                            .as_deref()
                            .expect("Stage2 verifier claim relation"),
                    ),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| generated_stage2::Stage2SumcheckBatchPlan {
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
                .map(|plan| generated_stage2::Stage2SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    relation: leak_str(
                        plan.relation
                            .as_deref()
                            .expect("Stage2 verifier driver relation"),
                    ),
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
                .map(|plan| generated_stage2::Stage2SumcheckInstanceResultPlan {
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
                .map(|plan| generated_stage2::Stage2SumcheckEvalPlan {
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
                .map(|plan| generated_stage2::Stage2PointSlicePlan {
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
                .map(|plan| generated_stage2::Stage2PointConcatPlan {
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
                .map(|plan| generated_stage2::Stage2OpeningClaimPlan {
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
                .map(|plan| generated_stage2::Stage2OpeningBatchPlan {
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

fn leak_generated_stage2_product_uniskip_verifier_program(
    program: &CompilerStage2CpuProgram,
) -> &'static generated_stage2::Stage2VerifierProgramPlan {
    let transcript_squeeze = program
        .transcript_squeezes
        .iter()
        .find(|plan| plan.symbol == "stage2.product_virtual.tau_high")
        .expect("product tau squeeze");
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

    Box::leak(Box::new(generated_stage2::Stage2VerifierProgramPlan {
        params: generated_stage2::Stage2Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(vec![
            generated_stage2::Stage2ProgramStepPlan {
                kind: "transcript_squeeze",
                symbol: leak_str(&transcript_squeeze.symbol),
            },
            generated_stage2::Stage2ProgramStepPlan {
                kind: "sumcheck_driver",
                symbol: leak_str(&driver.symbol),
            },
        ]),
        transcript_squeezes: leak_slice(vec![generated_stage2::Stage2TranscriptSqueezePlan {
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
                .map(|plan| generated_stage2::Stage2OpeningInputPlan {
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
        field_constants: &[] as &[generated_stage2::Stage2FieldConstantPlan],
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .filter(|plan| plan.symbol.starts_with("stage2.product_virtual.uniskip."))
                .map(|plan| generated_stage2::Stage2FieldExprPlan {
                    symbol: leak_str(&plan.symbol),
                    kind: leak_str(&plan.kind),
                    formula: leak_str(&plan.formula),
                    operand_names: leak_str_slice(&plan.operand_names),
                    operands: leak_str_slice(&plan.operands),
                })
                .collect(),
        ),
        claims: leak_slice(vec![generated_stage2::Stage2SumcheckClaimPlan {
            symbol: leak_str(&claim.symbol),
            stage: leak_str(&claim.stage),
            domain: leak_str(&claim.domain),
            num_rounds: claim.num_rounds,
            degree: claim.degree,
            claim: leak_str(&claim.claim),
            relation: leak_str(
                claim
                    .relation
                    .as_deref()
                    .expect("Stage2 product claim relation"),
            ),
            claim_value: leak_str(&claim.claim_value),
            input_openings: leak_str_slice(&claim.input_openings),
        }]),
        batches: leak_slice(vec![generated_stage2::Stage2SumcheckBatchPlan {
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
        drivers: leak_slice(vec![generated_stage2::Stage2SumcheckDriverPlan {
            symbol: leak_str(&driver.symbol),
            stage: leak_str(&driver.stage),
            proof_slot: leak_str(&driver.proof_slot),
            relation: leak_str(
                driver
                    .relation
                    .as_deref()
                    .expect("Stage2 product driver relation"),
            ),
            batch: leak_str(&driver.batch),
            policy: leak_str(&driver.policy),
            round_schedule: leak_usize_slice(&driver.round_schedule),
            claim_label: leak_str(&driver.claim_label),
            round_label: leak_str(&driver.round_label),
            num_rounds: driver.num_rounds,
            degree: driver.degree,
        }]),
        instance_results: leak_slice(vec![generated_stage2::Stage2SumcheckInstanceResultPlan {
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
        evals: leak_slice(vec![generated_stage2::Stage2SumcheckEvalPlan {
            symbol: leak_str(&eval.symbol),
            source: leak_str(&eval.source),
            name: leak_str(&eval.name),
            index: eval.index,
            oracle: leak_str(&eval.oracle),
        }]),
        point_slices: &[] as &[generated_stage2::Stage2PointSlicePlan],
        point_concats: &[] as &[generated_stage2::Stage2PointConcatPlan],
        opening_claims: leak_slice(vec![generated_stage2::Stage2OpeningClaimPlan {
            symbol: leak_str(&opening_claim.symbol),
            oracle: leak_str(&opening_claim.oracle),
            domain: leak_str(&opening_claim.domain),
            point_arity: opening_claim.point_arity,
            claim_kind: leak_str(&opening_claim.claim_kind),
            point_source: leak_str(&opening_claim.point_source),
            eval_source: leak_str(&opening_claim.eval_source),
        }]),
        opening_batches: &[] as &[generated_stage2::Stage2OpeningBatchPlan],
    }))
}

fn leak_generated_stage3_verifier_program(
    program: &CompilerStage3CpuProgram,
) -> &'static generated_stage3::Stage3VerifierProgramPlan {
    Box::leak(Box::new(generated_stage3::Stage3VerifierProgramPlan {
        params: generated_stage3::Stage3Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| generated_stage3::Stage3ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| generated_stage3::Stage3TranscriptSqueezePlan {
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
                .map(|plan| generated_stage3::Stage3OpeningInputPlan {
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
                .map(|plan| generated_stage3::Stage3FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| generated_stage3::Stage3FieldExprPlan {
                    symbol: leak_str(&plan.symbol),
                    kind: leak_str(&plan.kind),
                    formula: leak_str(&plan.formula),
                    operand_names: leak_str_slice(&plan.operand_names),
                    operands: leak_str_slice(&plan.operands),
                })
                .collect(),
        ),
        claims: leak_slice(
            program
                .claims
                .iter()
                .map(|plan| generated_stage3::Stage3SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    relation: leak_str(
                        plan.relation
                            .as_deref()
                            .expect("Stage3 verifier claim relation"),
                    ),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| generated_stage3::Stage3SumcheckBatchPlan {
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
                .map(|plan| generated_stage3::Stage3SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    relation: leak_str(
                        plan.relation
                            .as_deref()
                            .expect("Stage3 verifier driver relation"),
                    ),
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
                .map(|plan| generated_stage3::Stage3SumcheckInstanceResultPlan {
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
                .map(|plan| generated_stage3::Stage3SumcheckEvalPlan {
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
                .map(|plan| generated_stage3::Stage3PointSlicePlan {
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
                .map(|plan| generated_stage3::Stage3PointConcatPlan {
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
                .map(|plan| generated_stage3::Stage3OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| generated_stage3::Stage3OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| generated_stage3::Stage3OpeningBatchPlan {
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

fn leak_generated_stage4_verifier_program(
    program: &CompilerStage4CpuProgram,
) -> &'static generated_stage4::Stage4VerifierProgramPlan {
    Box::leak(Box::new(generated_stage4::Stage4VerifierProgramPlan {
        role: role_name(&program.role),
        params: generated_stage4::Stage4Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| generated_stage4::Stage4ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| generated_stage4::Stage4TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| generated_stage4::Stage4TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| generated_stage4::Stage4OpeningInputPlan {
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
                .map(|plan| generated_stage4::Stage4FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| generated_stage4::Stage4FieldExprPlan {
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
                .map(|plan| generated_stage4::Stage4KernelPlan {
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
                .map(|plan| generated_stage4::Stage4SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| generated_stage4::Stage4SumcheckBatchPlan {
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
                .map(|plan| generated_stage4::Stage4SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| generated_stage4::Stage4SumcheckInstanceResultPlan {
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
                .map(|plan| generated_stage4::Stage4SumcheckEvalPlan {
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
                .map(|plan| generated_stage4::Stage4PointSlicePlan {
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
                .map(|plan| generated_stage4::Stage4PointConcatPlan {
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
                .map(|plan| generated_stage4::Stage4OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| generated_stage4::Stage4OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| generated_stage4::Stage4OpeningBatchPlan {
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

fn leak_generated_stage5_verifier_program(
    program: &CompilerStage5CpuProgram,
) -> &'static generated_stage5::Stage5VerifierProgramPlan {
    Box::leak(Box::new(generated_stage5::Stage5VerifierProgramPlan {
        role: role_name(&program.role),
        params: generated_stage5::Stage5Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| generated_stage5::Stage5ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| generated_stage5::Stage5TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| generated_stage5::Stage5TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| generated_stage5::Stage5OpeningInputPlan {
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
                .map(|plan| generated_stage5::Stage5FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| generated_stage5::Stage5FieldExprPlan {
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
                .map(|plan| generated_stage5::Stage5KernelPlan {
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
                .map(|plan| generated_stage5::Stage5SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| generated_stage5::Stage5SumcheckBatchPlan {
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
                .map(|plan| generated_stage5::Stage5SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| generated_stage5::Stage5SumcheckInstanceResultPlan {
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
                .map(|plan| generated_stage5::Stage5SumcheckEvalPlan {
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
                .map(|plan| generated_stage5::Stage5PointSlicePlan {
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
                .map(|plan| generated_stage5::Stage5PointConcatPlan {
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
                .map(|plan| generated_stage5::Stage5OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| generated_stage5::Stage5OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| generated_stage5::Stage5OpeningBatchPlan {
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

fn leak_generated_stage6_verifier_program(
    program: &CompilerStage6CpuProgram,
) -> &'static generated_stage6::Stage6VerifierProgramPlan {
    Box::leak(Box::new(generated_stage6::Stage6VerifierProgramPlan {
        role: role_name(&program.role),
        params: generated_stage6::Stage6Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| generated_stage6::Stage6ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| generated_stage6::Stage6TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| generated_stage6::Stage6TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| generated_stage6::Stage6OpeningInputPlan {
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
                .map(|plan| generated_stage6::Stage6FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| generated_stage6::Stage6FieldExprPlan {
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
                .map(|plan| generated_stage6::Stage6KernelPlan {
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
                .map(|plan| generated_stage6::Stage6SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| generated_stage6::Stage6SumcheckBatchPlan {
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
                .map(|plan| generated_stage6::Stage6SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| generated_stage6::Stage6SumcheckInstanceResultPlan {
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
                .map(|plan| generated_stage6::Stage6SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_zeros: leak_slice(
            program
                .point_zeros
                .iter()
                .map(|plan| generated_stage6::Stage6PointZeroPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    arity: plan.arity,
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| generated_stage6::Stage6PointSlicePlan {
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
                .map(|plan| generated_stage6::Stage6PointConcatPlan {
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
                .map(|plan| generated_stage6::Stage6OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| generated_stage6::Stage6OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| generated_stage6::Stage6OpeningBatchPlan {
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

fn leak_generated_stage7_verifier_program(
    program: &CompilerStage7CpuProgram,
) -> &'static generated_stage7::Stage7VerifierProgramPlan {
    Box::leak(Box::new(generated_stage7::Stage7VerifierProgramPlan {
        role: role_name(&program.role),
        params: generated_stage7::Stage7Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| generated_stage7::Stage7ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| generated_stage7::Stage7TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| generated_stage7::Stage7TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| generated_stage7::Stage7OpeningInputPlan {
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
                .map(|plan| generated_stage7::Stage7FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| generated_stage7::Stage7FieldExprPlan {
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
                .map(|plan| generated_stage7::Stage7KernelPlan {
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
                .map(|plan| generated_stage7::Stage7SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| generated_stage7::Stage7SumcheckBatchPlan {
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
                .map(|plan| generated_stage7::Stage7SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| generated_stage7::Stage7SumcheckInstanceResultPlan {
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
                .map(|plan| generated_stage7::Stage7SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_zeros: leak_slice(
            program
                .point_zeros
                .iter()
                .map(|plan| generated_stage7::Stage7PointZeroPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    arity: plan.arity,
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| generated_stage7::Stage7PointSlicePlan {
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
                .map(|plan| generated_stage7::Stage7PointConcatPlan {
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
                .map(|plan| generated_stage7::Stage7OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| generated_stage7::Stage7OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| generated_stage7::Stage7OpeningBatchPlan {
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

fn leak_generated_stage8_prover_program(
    program: &CompilerStage8CpuProgram,
) -> &'static generated_prover_stage8::Stage8EvaluationProgramPlan {
    let evaluation_point_source = program
        .opening_inputs
        .iter()
        .find(|input| input.symbol == "stage8.evaluation.point_source")
        .expect("stage8 evaluation point source exists");
    Box::leak(Box::new(
        generated_prover_stage8::Stage8EvaluationProgramPlan {
            role: role_name(&program.role),
            function: leak_str(&program.function),
            params: generated_prover_stage8::Stage8Params {
                field: leak_str(&program.params.field),
                pcs: leak_str(&program.params.pcs),
                transcript: leak_str(&program.params.transcript),
            },
            evaluation_point_source: generated_prover_stage8::Stage8OpeningInputPlan {
                symbol: leak_str(&evaluation_point_source.symbol),
                source_stage: leak_str(&evaluation_point_source.source_stage),
                source_claim: leak_str(&evaluation_point_source.source_claim),
                oracle: leak_str(&evaluation_point_source.oracle),
                domain: leak_str(&evaluation_point_source.domain),
                point_arity: evaluation_point_source.point_arity,
                claim_kind: leak_str(&evaluation_point_source.claim_kind),
            },
            opening_inputs: leak_slice(
                program
                    .opening_inputs
                    .iter()
                    .map(|plan| generated_prover_stage8::Stage8OpeningInputPlan {
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
            opening_claims: leak_slice(
                program
                    .opening_claims
                    .iter()
                    .map(|plan| generated_prover_stage8::Stage8OpeningClaimPlan {
                        symbol: leak_str(&plan.symbol),
                        oracle: leak_str(&plan.oracle),
                        family: leak_str(&plan.family),
                        domain: leak_str(&plan.domain),
                        point_arity: plan.point_arity,
                        point_source: leak_str(&plan.point_source),
                        eval_source: leak_str(&plan.eval_source),
                        source_stage: leak_str(&plan.source_stage),
                        source_claim: leak_str(&plan.source_claim),
                    })
                    .collect(),
            ),
            opening_batch: generated_prover_stage8::Stage8OpeningBatchPlan {
                symbol: leak_str(&program.opening_batches[0].symbol),
                proof_slot: leak_str(&program.opening_batches[0].proof_slot),
                policy: leak_str(&program.opening_batches[0].policy),
                count: program.opening_batches[0].count,
                ordered_claims: leak_str_slice(&program.opening_batches[0].ordered_claims),
            },
            pcs_proof: generated_prover_stage8::Stage8PcsProofPlan {
                symbol: leak_str(&program.pcs_proofs[0].symbol),
                mode: leak_str(&program.pcs_proofs[0].mode),
                pcs: leak_str(&program.pcs_proofs[0].pcs),
                proof_slot: leak_str(&program.pcs_proofs[0].proof_slot),
                transcript_label: leak_str(&program.pcs_proofs[0].transcript_label),
                batch: leak_str(&program.pcs_proofs[0].batch),
            },
        },
    ))
}

fn leak_generated_stage8_verifier_program(
    program: &CompilerStage8CpuProgram,
) -> &'static generated_stage8::Stage8EvaluationProgramPlan {
    let evaluation_point_source = program
        .opening_inputs
        .iter()
        .find(|input| input.symbol == "stage8.evaluation.point_source")
        .expect("stage8 evaluation point source exists");
    Box::leak(Box::new(generated_stage8::Stage8EvaluationProgramPlan {
        role: role_name(&program.role),
        function: leak_str(&program.function),
        params: generated_stage8::Stage8Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        evaluation_point_source: generated_stage8::Stage8OpeningInputPlan {
            symbol: leak_str(&evaluation_point_source.symbol),
            source_stage: leak_str(&evaluation_point_source.source_stage),
            source_claim: leak_str(&evaluation_point_source.source_claim),
            oracle: leak_str(&evaluation_point_source.oracle),
            domain: leak_str(&evaluation_point_source.domain),
            point_arity: evaluation_point_source.point_arity,
            claim_kind: leak_str(&evaluation_point_source.claim_kind),
        },
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| generated_stage8::Stage8OpeningInputPlan {
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
        opening_claims: leak_slice(
            program
                .opening_claims
                .iter()
                .map(|plan| generated_stage8::Stage8OpeningClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    oracle: leak_str(&plan.oracle),
                    family: leak_str(&plan.family),
                    domain: leak_str(&plan.domain),
                    point_arity: plan.point_arity,
                    point_source: leak_str(&plan.point_source),
                    eval_source: leak_str(&plan.eval_source),
                    source_stage: leak_str(&plan.source_stage),
                    source_claim: leak_str(&plan.source_claim),
                })
                .collect(),
        ),
        opening_batch: generated_stage8::Stage8OpeningBatchPlan {
            symbol: leak_str(&program.opening_batches[0].symbol),
            proof_slot: leak_str(&program.opening_batches[0].proof_slot),
            policy: leak_str(&program.opening_batches[0].policy),
            count: program.opening_batches[0].count,
            ordered_claims: leak_str_slice(&program.opening_batches[0].ordered_claims),
        },
        pcs_proof: generated_stage8::Stage8PcsProofPlan {
            symbol: leak_str(&program.pcs_proofs[0].symbol),
            mode: leak_str(&program.pcs_proofs[0].mode),
            pcs: leak_str(&program.pcs_proofs[0].pcs),
            proof_slot: leak_str(&program.pcs_proofs[0].proof_slot),
            transcript_label: leak_str(&program.pcs_proofs[0].transcript_label),
            batch: leak_str(&program.pcs_proofs[0].batch),
        },
    }))
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::Prover => "prover",
        Role::Verifier => "verifier",
    }
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
    let core_bytecode_for_bolt = core_bytecode.clone();
    let (_, trace, _, host_io_device) = core_program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        host_io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        entry_address,
    )
    .expect("shared preprocessing");
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

    let mut padded_trace = trace.clone();
    padded_trace.resize(proof.trace_length, jolt_trace::Cycle::NoOp);
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
    let bytecode = BytecodePreprocessing::preprocess(core_bytecode_for_bolt, entry_address);
    let stage6_bytecode_entries = generated_stage6_bytecode_entries(&bytecode);
    let stage6_entry_bytecode_index = bytecode.entry_bytecode_index();
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), proof.trace_length);
    let (cycle_inputs, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        proof.trace_length,
        &bytecode,
        &host_io_device.memory_layout,
        r1cs_key.num_vars_padded,
    );
    let rv64_cycles = stage1_rv64_cycles(&trace, proof.trace_length, &bytecode);
    let stage3_cycles = stage3_cycles(&trace, proof.trace_length, &bytecode);
    let stage4_register_accesses = stage4_register_accesses(&trace, proof.trace_length);
    let stage5_lookup_indices = padded_trace
        .iter()
        .map(|cycle| LookupQuery::<XLEN>::to_lookup_index(cycle))
        .collect();
    let stage5_lookup_table_indices = padded_trace
        .iter()
        .map(|cycle| {
            InstructionLookup::<XLEN>::lookup_table(cycle)
                .map(|table| CoreLookupTables::<XLEN>::enum_index(&table))
        })
        .collect();
    let stage5_is_interleaved_operands = padded_trace
        .iter()
        .map(|cycle| {
            cycle
                .instruction()
                .circuit_flags()
                .is_interleaved_operands()
        })
        .collect();
    let core_rv64_cycles = (0..proof.trace_length)
        .map(|index| {
            let row = CoreR1CSCycleInputs::from_trace::<CoreFr>(
                &prover_preprocessing.shared.bytecode,
                &padded_trace,
                index,
            );
            stage1_rv64_cycle_from_core_r1cs(&row)
        })
        .collect();

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
        rv64_cycles,
        core_rv64_cycles,
        product_virtual_cycles,
        instruction_lookup_cycles,
        stage3_cycles,
        stage4_register_accesses,
        stage5_lookup_indices,
        stage5_lookup_table_indices,
        stage5_is_interleaved_operands,
        padded_trace,
        stage6_bytecode_entries,
        stage6_entry_bytecode_index,
        ram_accesses,
        initial_ram_state,
        final_ram_state,
        ram_start_address,
        ram_output_layout,
        commitments,
    }
}

fn generated_stage6_verifier_data(
    fixture: &CoreMuldivCommitmentFixture,
) -> generated_stage6::Stage6VerifierData {
    generated_stage6::Stage6VerifierData {
        bytecode_read_raf: Some(generated_stage6::Stage6BytecodeReadRafData {
            entries: fixture.stage6_bytecode_entries.clone(),
            entry_bytecode_index: fixture.stage6_entry_bytecode_index,
            num_lookup_tables: CoreLookupTables::<XLEN>::COUNT,
        }),
    }
}

fn generated_stage6_bytecode_entries(
    bytecode: &BytecodePreprocessing,
) -> Vec<generated_stage6::Stage6BytecodeEntry> {
    bytecode
        .bytecode
        .iter()
        .map(|instruction| {
            let instr = instruction.normalize();
            let circuit_flags = instruction.circuit_flags();
            let instruction_flags = instruction.instruction_flags();
            let lookup_table = InstructionLookup::<XLEN>::lookup_table(instruction)
                .map(|table| CoreLookupTables::<XLEN>::enum_index(&table));
            generated_stage6::Stage6BytecodeEntry {
                address: Fr::from_u64(instr.address as u64),
                imm: Fr::from_i128(instr.operands.imm),
                circuit_flags: generated_stage6_circuit_flags(circuit_flags),
                rd: instr.operands.rd.map(usize::from),
                rs1: instr.operands.rs1.map(usize::from),
                rs2: instr.operands.rs2.map(usize::from),
                lookup_table,
                is_interleaved: circuit_flags.is_interleaved_operands(),
                is_branch: instruction_flags[InstructionFlags::Branch],
                left_is_rs1: instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
                left_is_pc: instruction_flags[InstructionFlags::LeftOperandIsPC],
                right_is_rs2: instruction_flags[InstructionFlags::RightOperandIsRs2Value],
                right_is_imm: instruction_flags[InstructionFlags::RightOperandIsImm],
                is_noop: instruction_flags[InstructionFlags::IsNoop],
            }
        })
        .collect()
}

fn generated_stage6_circuit_flags(flags: [bool; 14]) -> [bool; 14] {
    [
        flags[CircuitFlags::AddOperands],
        flags[CircuitFlags::SubtractOperands],
        flags[CircuitFlags::MultiplyOperands],
        flags[CircuitFlags::Load],
        flags[CircuitFlags::Store],
        flags[CircuitFlags::Jump],
        flags[CircuitFlags::WriteLookupOutputToRD],
        flags[CircuitFlags::VirtualInstruction],
        flags[CircuitFlags::Assert],
        flags[CircuitFlags::DoNotUpdateUnexpandedPC],
        flags[CircuitFlags::Advice],
        flags[CircuitFlags::IsCompressed],
        flags[CircuitFlags::IsFirstInSequence],
        flags[CircuitFlags::IsLastInSequence],
    ]
}

fn stage1_rv64_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
) -> Vec<Stage1Rv64Cycle> {
    (0..size)
        .map(|cycle| stage1_rv64_cycle(trace, cycle, bytecode))
        .collect()
}

fn stage3_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
) -> Vec<Stage3Cycle> {
    (0..size)
        .map(|cycle| stage3_cycle(trace.get(cycle).copied(), bytecode))
        .collect()
}

fn stage4_register_accesses<C: CycleRow>(trace: &[C], size: usize) -> Vec<Stage4RegisterAccess> {
    (0..size)
        .map(|cycle| stage4_register_access(trace.get(cycle).copied()))
        .collect()
}

fn stage4_register_access<C: CycleRow>(cycle: Option<C>) -> Stage4RegisterAccess {
    let Some(cycle) = cycle else {
        return Stage4RegisterAccess::default();
    };
    Stage4RegisterAccess {
        rs1: cycle.rs1_read().map(|(address, value)| Stage4RegisterRead {
            address: address as usize,
            value,
        }),
        rs2: cycle.rs2_read().map(|(address, value)| Stage4RegisterRead {
            address: address as usize,
            value,
        }),
        rd: cycle
            .rd_write()
            .map(|(address, pre_value, post_value)| Stage4RegisterWrite {
                address: address as usize,
                pre_value,
                post_value,
            }),
    }
}

fn stage3_cycle<C: CycleRow>(cycle: Option<C>, bytecode: &BytecodePreprocessing) -> Stage3Cycle {
    let Some(cycle) = cycle else {
        return Stage3Cycle::padding();
    };
    let circuit_flags = cycle.circuit_flags();
    let instruction_flags = cycle.instruction_flags();
    Stage3Cycle {
        unexpanded_pc: cycle.unexpanded_pc(),
        pc: bytecode.get_cycle_pc(&cycle) as u64,
        is_virtual: circuit_flags[TraceCircuitFlags::VirtualInstruction],
        is_first_in_sequence: circuit_flags[TraceCircuitFlags::IsFirstInSequence],
        is_noop: instruction_flags[TraceInstructionFlags::IsNoop],
        left_operand_is_rs1: instruction_flags[TraceInstructionFlags::LeftOperandIsRs1Value],
        rs1_value: cycle.rs1_read().map_or(0, |(_, value)| value),
        left_operand_is_pc: instruction_flags[TraceInstructionFlags::LeftOperandIsPC],
        right_operand_is_rs2: instruction_flags[TraceInstructionFlags::RightOperandIsRs2Value],
        rs2_value: cycle.rs2_read().map_or(0, |(_, value)| value),
        right_operand_is_imm: instruction_flags[TraceInstructionFlags::RightOperandIsImm],
        imm: cycle.imm(),
        rd_write_value: cycle.rd_write().map_or(0, |(_, _, post)| post),
    }
}

fn stage1_rv64_cycle<C: CycleRow>(
    trace: &[C],
    cycle_index: usize,
    bytecode: &BytecodePreprocessing,
) -> Stage1Rv64Cycle {
    let Some(cycle) = trace.get(cycle_index) else {
        return Stage1Rv64Cycle::padding();
    };
    let next = trace.get(cycle_index + 1);
    if cycle.is_noop() {
        let mut row = Stage1Rv64Cycle::padding();
        fill_next_rv64_fields(&mut row, next, bytecode);
        return row;
    }

    let flags_set = cycle.circuit_flags();
    let instruction_flags = cycle.instruction_flags();
    let left_input = if instruction_flags[TraceInstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if instruction_flags[TraceInstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0, |(_, value)| value)
    } else {
        0
    };
    let right_i128 = if instruction_flags[TraceInstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if instruction_flags[TraceInstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0, |(_, value)| value as i128)
    } else {
        0
    };
    let right_input = s64_from_i128(right_i128);
    let product = S64::from_u64(left_input).mul_trunc::<2, 2>(&S128::from_i128(right_i128));
    let lookup_output = cycle.lookup_output();
    let (left_lookup, right_lookup) =
        lookup_operands_raw(left_input, right_i128, product, flags_set, lookup_output);
    let next_is_noop = next.is_none_or(CycleRow::is_noop);
    let flags = stage1_rv64_flags(flags_set);

    let mut row = Stage1Rv64Cycle {
        left_input,
        right_input,
        product,
        left_lookup,
        right_lookup,
        lookup_output,
        rs1_read_value: cycle.rs1_read().map_or(0, |(_, value)| value),
        rs2_read_value: cycle.rs2_read().map_or(0, |(_, value)| value),
        rd_write_value: cycle.rd_write().map_or(0, |(_, _, post)| post),
        ram_addr: cycle.ram_access_address().unwrap_or(0),
        ram_read_value: cycle.ram_read_value().unwrap_or(0),
        ram_write_value: cycle.ram_write_value().unwrap_or(0),
        pc: bytecode.get_cycle_pc(cycle) as u64,
        next_pc: 0,
        unexpanded_pc: cycle.unexpanded_pc(),
        next_unexpanded_pc: 0,
        imm: s64_from_i128(cycle.imm()),
        flags,
        should_jump: flags_set[TraceCircuitFlags::Jump] && !next_is_noop,
        should_branch: instruction_flags[TraceInstructionFlags::Branch] && lookup_output == 1,
        next_is_virtual: false,
        next_is_first_in_sequence: false,
    };
    fill_next_rv64_fields(&mut row, next, bytecode);
    row
}

fn stage1_rv64_cycle_from_core_r1cs(row: &CoreR1CSCycleInputs) -> Stage1Rv64Cycle {
    Stage1Rv64Cycle {
        left_input: row.left_input,
        right_input: s64_from_i128(row.right_input.to_i128()),
        product: S128::from_u128_and_sign(row.product.magnitude_as_u128(), row.product.is_positive),
        left_lookup: row.left_lookup,
        right_lookup: row.right_lookup,
        lookup_output: row.lookup_output,
        rs1_read_value: row.rs1_read_value,
        rs2_read_value: row.rs2_read_value,
        rd_write_value: row.rd_write_value,
        ram_addr: row.ram_addr,
        ram_read_value: row.ram_read_value,
        ram_write_value: row.ram_write_value,
        pc: row.pc,
        next_pc: row.next_pc,
        unexpanded_pc: row.unexpanded_pc,
        next_unexpanded_pc: row.next_unexpanded_pc,
        imm: s64_from_i128(row.imm.to_i128()),
        flags: [
            row.flags[CircuitFlags::AddOperands],
            row.flags[CircuitFlags::SubtractOperands],
            row.flags[CircuitFlags::MultiplyOperands],
            row.flags[CircuitFlags::Load],
            row.flags[CircuitFlags::Store],
            row.flags[CircuitFlags::Jump],
            row.flags[CircuitFlags::WriteLookupOutputToRD],
            row.flags[CircuitFlags::VirtualInstruction],
            row.flags[CircuitFlags::Assert],
            row.flags[CircuitFlags::DoNotUpdateUnexpandedPC],
            row.flags[CircuitFlags::Advice],
            row.flags[CircuitFlags::IsCompressed],
            row.flags[CircuitFlags::IsFirstInSequence],
            row.flags[CircuitFlags::IsLastInSequence],
        ],
        should_jump: row.should_jump,
        should_branch: row.should_branch,
        next_is_virtual: row.next_is_virtual,
        next_is_first_in_sequence: row.next_is_first_in_sequence,
    }
}

fn assert_rv64_cycles_match_core(actual: &[Stage1Rv64Cycle], expected: &[Stage1Rv64Cycle]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "RV64 cycle count differs from core"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_rv64_cycle_matches_core(index, actual, expected);
    }
}

fn assert_rv64_cycle_matches_core(
    index: usize,
    actual: &Stage1Rv64Cycle,
    expected: &Stage1Rv64Cycle,
) {
    macro_rules! check_field {
        ($field:ident) => {
            if actual.$field != expected.$field {
                panic!(
                    "RV64 cycle {index} field `{}` differs from core: actual={:?} expected={:?}",
                    stringify!($field),
                    actual.$field,
                    expected.$field
                );
            }
        };
    }

    check_field!(left_input);
    check_field!(right_input);
    check_field!(product);
    check_field!(left_lookup);
    check_field!(right_lookup);
    check_field!(lookup_output);
    check_field!(rs1_read_value);
    check_field!(rs2_read_value);
    check_field!(rd_write_value);
    check_field!(ram_addr);
    check_field!(ram_read_value);
    check_field!(ram_write_value);
    check_field!(pc);
    check_field!(next_pc);
    check_field!(unexpanded_pc);
    check_field!(next_unexpanded_pc);
    check_field!(imm);
    check_field!(should_jump);
    check_field!(should_branch);
    check_field!(next_is_virtual);
    check_field!(next_is_first_in_sequence);

    for (flag_index, (actual, expected)) in actual.flags.iter().zip(expected.flags).enumerate() {
        if *actual != expected {
            panic!(
                "RV64 cycle {index} flag {flag_index} differs from core: actual={actual:?} expected={expected:?}"
            );
        }
    }
}

fn fill_next_rv64_fields<C: CycleRow>(
    row: &mut Stage1Rv64Cycle,
    next: Option<&C>,
    bytecode: &BytecodePreprocessing,
) {
    if let Some(next_cycle) = next {
        row.next_pc = bytecode.get_cycle_pc(next_cycle) as u64;
        row.next_unexpanded_pc = next_cycle.unexpanded_pc();
        let next_flags = next_cycle.circuit_flags();
        row.next_is_virtual = next_flags[TraceCircuitFlags::VirtualInstruction];
        row.next_is_first_in_sequence = next_flags[TraceCircuitFlags::IsFirstInSequence];
    }
}

fn stage1_rv64_flags(flags: jolt_trace::CircuitFlagSet) -> [bool; jolt_trace::NUM_CIRCUIT_FLAGS] {
    [
        flags[TraceCircuitFlags::AddOperands],
        flags[TraceCircuitFlags::SubtractOperands],
        flags[TraceCircuitFlags::MultiplyOperands],
        flags[TraceCircuitFlags::Load],
        flags[TraceCircuitFlags::Store],
        flags[TraceCircuitFlags::Jump],
        flags[TraceCircuitFlags::WriteLookupOutputToRD],
        flags[TraceCircuitFlags::VirtualInstruction],
        flags[TraceCircuitFlags::Assert],
        flags[TraceCircuitFlags::DoNotUpdateUnexpandedPC],
        flags[TraceCircuitFlags::Advice],
        flags[TraceCircuitFlags::IsCompressed],
        flags[TraceCircuitFlags::IsFirstInSequence],
        flags[TraceCircuitFlags::IsLastInSequence],
    ]
}

fn s64_from_i128(value: i128) -> S64 {
    let magnitude = value.unsigned_abs();
    assert!(magnitude <= u64::MAX as u128, "S64 input overflow");
    S64::from_u64_with_sign(magnitude as u64, value >= 0)
}

fn lookup_operands_raw(
    left: u64,
    right: i128,
    product: S128,
    flags: jolt_trace::CircuitFlagSet,
    lookup_output: u64,
) -> (u64, u128) {
    if flags[TraceCircuitFlags::AddOperands] {
        (0, (left as i128 + right) as u128)
    } else if flags[TraceCircuitFlags::SubtractOperands] {
        (0, (left as i128 - right + (1i128 << 64)) as u128)
    } else if flags[TraceCircuitFlags::MultiplyOperands] {
        (0, product.magnitude_as_u128())
    } else if flags[TraceCircuitFlags::Advice] {
        (0, lookup_output as u128)
    } else {
        (left, right as u128)
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

fn append_bolt_preamble<T>(transcript: &mut T, fixture: &CoreMuldivCommitmentFixture)
where
    T: Transcript<Challenge = Fr>,
{
    let program_io = &fixture.io;
    let preprocessing_digest = fixture.verifier_preprocessing.shared.digest();
    append_bytes(transcript, b"preprocessing_digest", &preprocessing_digest);
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
    append_u64(transcript, b"ram_K", fixture.proof.ram_K as u64);
    append_u64(
        transcript,
        b"trace_length",
        fixture.proof.trace_length as u64,
    );
    append_u64(transcript, b"entry_address", fixture.entry_address);
    append_u64(
        transcript,
        b"ram_rw_phase1_num_rounds",
        fixture.proof.rw_config.ram_rw_phase1_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"ram_rw_phase2_num_rounds",
        fixture.proof.rw_config.ram_rw_phase2_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"registers_rw_phase1_num_rounds",
        fixture.proof.rw_config.registers_rw_phase1_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"registers_rw_phase2_num_rounds",
        fixture.proof.rw_config.registers_rw_phase2_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"log_k_chunk",
        fixture.proof.one_hot_config.log_k_chunk as u64,
    );
    append_u64(
        transcript,
        b"lookups_ra_virtual_log_k_chunk",
        fixture.proof.one_hot_config.lookups_ra_virtual_log_k_chunk as u64,
    );
    append_u64(transcript, b"dory_layout", fixture.proof.dory_layout as u64);
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
    assert_core_stage1_tau_matches_bolt(fixture, artifacts);
    assert_core_stage1_uniskip_proof_matches_bolt(&fixture.proof, &artifacts.sumchecks[0]);
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

fn assert_core_stage1_tau_matches_bolt(
    fixture: &CoreMuldivCommitmentFixture,
    artifacts: &Stage1ExecutionArtifacts<Fr>,
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
    let (params, _) = jolt_core::zkvm::spartan::verify_stage1_uni_skip(
        &fixture.proof.stage1_uni_skip_first_round_proof,
        &verifier.spartan_key,
        &mut verifier.opening_accumulator,
        &mut verifier.transcript,
    )
    .expect("core Stage 1 uni-skip verifies");
    let bolt_tau = artifacts
        .challenge_vectors
        .iter()
        .find(|vector| vector.symbol == "stage1.tau")
        .expect("Bolt stage1 tau");
    let core_tau = params
        .tau
        .into_iter()
        .map(|challenge| Fr::from(CoreFr::from(challenge)))
        .collect::<Vec<_>>();
    assert_eq!(bolt_tau.values, core_tau, "Stage 1 tau mismatch");
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
    assert_core_stage2_sumcheck_proof_matches_bolt(&fixture.proof, &stage2_artifacts.sumchecks[1]);
    assert_core_stage2_opening_claims_match_bolt(&fixture.proof, stage2_artifacts);

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

fn assert_core_accepts_bolt_stage3(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);
    assert_core_stage3_sumcheck_proof_matches_bolt(&fixture.proof, &stage3_artifacts.sumchecks[0]);
    assert_core_stage3_opening_claims_match_bolt(&fixture.proof, stage3_artifacts);

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
    let _ = verifier.verify_stage3().expect("core accepts Bolt Stage 3");
}

fn assert_core_accepts_bolt_stage4(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage4_sumcheck_proof =
        to_core_sumcheck_proof(&stage4_artifacts.sumchecks[0].proof.round_polynomials);

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
    let _ = verifier.verify_stage3().expect("core accepts Bolt Stage 3");
    let _ = verifier.verify_stage4().expect("core accepts Bolt Stage 4");
}

fn assert_core_stage3_opening_claims_match_bolt(
    proof: &CoreProof,
    artifacts: &Stage3ExecutionArtifacts<Fr>,
) {
    let expected = [
        (
            "stage3.spartan_shift.eval.UnexpandedPC",
            OpeningId::virt(VirtualPolynomial::UnexpandedPC, SumcheckId::SpartanShift),
        ),
        (
            "stage3.spartan_shift.eval.PC",
            OpeningId::virt(VirtualPolynomial::PC, SumcheckId::SpartanShift),
        ),
        (
            "stage3.spartan_shift.eval.OpFlagVirtualInstruction",
            OpeningId::virt(
                VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
                SumcheckId::SpartanShift,
            ),
        ),
        (
            "stage3.spartan_shift.eval.OpFlagIsFirstInSequence",
            OpeningId::virt(
                VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
                SumcheckId::SpartanShift,
            ),
        ),
        (
            "stage3.spartan_shift.eval.InstructionFlagIsNoop",
            OpeningId::virt(
                VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
                SumcheckId::SpartanShift,
            ),
        ),
        (
            "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value",
            OpeningId::virt(
                VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.instruction_input.eval.Rs1Value",
            OpeningId::virt(
                VirtualPolynomial::Rs1Value,
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC",
            OpeningId::virt(
                VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.instruction_input.eval.UnexpandedPC",
            OpeningId::virt(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value",
            OpeningId::virt(
                VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.instruction_input.eval.Rs2Value",
            OpeningId::virt(
                VirtualPolynomial::Rs2Value,
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm",
            OpeningId::virt(
                VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.instruction_input.eval.Imm",
            OpeningId::virt(
                VirtualPolynomial::Imm,
                SumcheckId::InstructionInputVirtualization,
            ),
        ),
        (
            "stage3.registers_claim_reduction.eval.RdWriteValue",
            OpeningId::virt(
                VirtualPolynomial::RdWriteValue,
                SumcheckId::RegistersClaimReduction,
            ),
        ),
        (
            "stage3.registers_claim_reduction.eval.Rs1Value",
            OpeningId::virt(
                VirtualPolynomial::Rs1Value,
                SumcheckId::RegistersClaimReduction,
            ),
        ),
        (
            "stage3.registers_claim_reduction.eval.Rs2Value",
            OpeningId::virt(
                VirtualPolynomial::Rs2Value,
                SumcheckId::RegistersClaimReduction,
            ),
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
                    "Stage 3 opening claim mismatch for {}",
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
    generated_stage2_verifier_plan: &'static generated_stage2::Stage2VerifierProgramPlan,
    stage1_artifacts: &'a Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &'a Stage2ExecutionArtifacts<Fr>,
    prover_transcript: &'a CheckpointTranscript<jolt_transcript::Blake2bTranscript<Fr>>,
}

fn assert_bolt_chain_verifier_accepts_stage2_product_uniskip(
    input: BoltStage2ChainVerifierInput<'_>,
) {
    let mut verifier_transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut verifier_transcript, input.fixture);
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

    let assert_stage2_product_tamper_rejected =
        |tampered_stage2_artifacts: Stage2ExecutionArtifacts<Fr>, message: &str| {
            let mut tamper_transcript = CheckpointTranscript::<
                jolt_transcript::Blake2bTranscript<Fr>,
            >::new(TRANSCRIPT_LABEL);
            append_bolt_preamble(&mut tamper_transcript, input.fixture);
            append_bolt_commitments_to_transcript(
                &mut tamper_transcript,
                &input.commitment_verifier_trace.records,
                &input.commitment_verifier_trace.commitments,
                &input.commitment_verifier_program.transcript_steps,
            );
            let mut tamper_stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
            let tamper_verified_stage1 = execute_stage1_program(
                input.stage1_prover_plan,
                Stage1ExecutionMode::Verifier,
                &mut tamper_stage1_verifier,
                &mut tamper_transcript,
            )
            .expect("Bolt Stage 1 verifier accepts before Stage 2 product uni-skip tamper");
            let tamper_stage2_openings = stage2_product_opening_inputs(&tamper_verified_stage1);
            let tampered_stage2_proof = Stage2Proof::from(tampered_stage2_artifacts.clone());
            let mut tamper_stage2_verifier =
                Stage2VerifierKernelExecutor::new(&tampered_stage2_proof, &tamper_stage2_openings);
            let tamper_result = execute_stage2_program(
                input.stage2_prover_plan,
                Stage2ExecutionMode::Verifier,
                &mut tamper_stage2_verifier,
                &mut tamper_transcript,
            );
            assert!(tamper_result.is_err(), "{message}");

            let mut generated_tamper_transcript = CheckpointTranscript::<
                jolt_transcript::Blake2bTranscript<Fr>,
            >::new(TRANSCRIPT_LABEL);
            append_bolt_preamble(&mut generated_tamper_transcript, input.fixture);
            append_bolt_commitments_to_transcript(
                &mut generated_tamper_transcript,
                &input.commitment_verifier_trace.records,
                &input.commitment_verifier_trace.commitments,
                &input.commitment_verifier_program.transcript_steps,
            );
            let mut generated_tamper_stage1_verifier =
                Stage1VerifierKernelExecutor::new(&stage1_proof);
            let generated_tamper_verified_stage1 = execute_stage1_program(
                input.stage1_prover_plan,
                Stage1ExecutionMode::Verifier,
                &mut generated_tamper_stage1_verifier,
                &mut generated_tamper_transcript,
            )
            .expect("Bolt Stage 1 verifier accepts before generated Stage 2 product tamper");
            let generated_tamper_stage2_openings = generated_stage2_opening_inputs(
                &stage2_product_opening_inputs(&generated_tamper_verified_stage1),
            );
            let generated_tampered_stage2_proof =
                to_generated_stage2_proof(&tampered_stage2_artifacts);
            let generated_tamper_result = generated_stage2::verify_stage2_with_program(
                input.generated_stage2_verifier_plan,
                &generated_tampered_stage2_proof,
                &generated_tamper_stage2_openings,
                None,
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");
        };

    let mut tampered_sumcheck = input.stage2_artifacts.clone();
    let tampered_poly = &mut tampered_sumcheck.sumchecks[0].proof.round_polynomials[0];
    let mut tampered_coefficients = tampered_poly.coefficients().to_vec();
    tampered_coefficients[0] += Fr::from_u64(1);
    *tampered_poly = UnivariatePoly::new(tampered_coefficients);
    assert_stage2_product_tamper_rejected(
        tampered_sumcheck,
        "Bolt Stage 2 verifier accepted a tampered product uni-skip coefficient",
    );

    let mut tampered_eval = input.stage2_artifacts.clone();
    tampered_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    assert_stage2_product_tamper_rejected(
        tampered_eval,
        "Bolt Stage 2 verifier accepted a tampered product uni-skip opening evaluation",
    );

    let mut tampered_point = input.stage2_artifacts.clone();
    tampered_point.sumchecks[0].point[0] += Fr::from_u64(1);
    assert_stage2_product_tamper_rejected(
        tampered_point,
        "Bolt Stage 2 verifier accepted a tampered product uni-skip point",
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

fn stage3_opening_inputs(
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
) -> Vec<Stage3OpeningInputValue<Fr>> {
    let mut inputs = Vec::new();
    let trace_point_arity = stage1_artifacts
        .opening_value("stage1.outer_remaining.opening.NextUnexpandedPC")
        .expect("Stage 1 NextUnexpandedPC opening")
        .point
        .len();
    for (oracle, symbol) in [
        ("NextUnexpandedPC", "stage3.input.stage1.NextUnexpandedPC"),
        ("NextPC", "stage3.input.stage1.NextPC"),
        ("NextIsVirtual", "stage3.input.stage1.NextIsVirtual"),
        (
            "NextIsFirstInSequence",
            "stage3.input.stage1.NextIsFirstInSequence",
        ),
        ("RdWriteValue", "stage3.input.stage1.RdWriteValue"),
        ("Rs1Value", "stage3.input.stage1.Rs1Value"),
        ("Rs2Value", "stage3.input.stage1.Rs2Value"),
    ] {
        let source_claim = format!("stage1.outer_remaining.opening.{oracle}");
        let opening = stage1_artifacts
            .opening_value(&source_claim)
            .unwrap_or_else(|| panic!("missing Stage 1 opening {source_claim}"));
        inputs.push(Stage3OpeningInputValue {
            symbol,
            point: opening.point.clone(),
            eval: opening.eval,
        });
    }

    for (eval_name, symbol) in [
        (
            "stage2.product_virtual.remainder.eval.NextIsNoop",
            "stage3.input.stage2.product_virtual.NextIsNoop",
        ),
        (
            "stage2.product_virtual.remainder.eval.LeftInstructionInput",
            "stage3.input.stage2.product_virtual.LeftInstructionInput",
        ),
        (
            "stage2.product_virtual.remainder.eval.RightInstructionInput",
            "stage3.input.stage2.product_virtual.RightInstructionInput",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            "stage3.input.stage2.instruction_lookup.LeftInstructionInput",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            "stage3.input.stage2.instruction_lookup.RightInstructionInput",
        ),
    ] {
        inputs.push(stage3_stage2_opening_input(
            stage2_artifacts,
            eval_name,
            symbol,
            trace_point_arity,
        ));
    }
    inputs
}

fn stage3_stage2_opening_input(
    artifacts: &Stage2ExecutionArtifacts<Fr>,
    eval_name: &'static str,
    symbol: &'static str,
    trace_point_arity: usize,
) -> Stage3OpeningInputValue<Fr> {
    let output = artifacts
        .sumchecks
        .iter()
        .find(|output| output.evals.iter().any(|eval| eval.name == eval_name))
        .unwrap_or_else(|| panic!("missing Stage 2 output for {eval_name}"));
    let eval = output
        .evals
        .iter()
        .find(|eval| eval.name == eval_name)
        .unwrap_or_else(|| panic!("missing Stage 2 eval {eval_name}"));
    let point_start = output
        .point
        .len()
        .checked_sub(trace_point_arity)
        .unwrap_or_else(|| panic!("Stage 2 point is shorter than trace point for {eval_name}"));
    let mut point = output.point[point_start..].to_vec();
    // Core caches these trace-domain openings in big-endian order.
    point.reverse();
    Stage3OpeningInputValue {
        symbol,
        point,
        eval: eval.value,
    }
}

fn stage4_opening_inputs(
    params: &JoltProtocolParams,
    initial_ram_state: &[u64],
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
) -> Vec<Stage4OpeningInputValue<Fr>> {
    let stage2 = stage2_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage2.sumcheck")
        .expect("Stage 2 batched output");
    let stage3 = stage3_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage3.sumcheck")
        .expect("Stage 3 batched output");
    let stage3_point = reverse_point(&stage3.point);
    let ram_val_point = reverse_point(&stage2.point);
    let ram_final_point = reversed_suffix(&stage2.point, params.log_k_ram);
    let mut inputs = Vec::new();
    for (eval_name, symbol) in [
        (
            "stage3.registers_claim_reduction.eval.RdWriteValue",
            "stage4.input.stage3.registers.RdWriteValue",
        ),
        (
            "stage3.registers_claim_reduction.eval.Rs1Value",
            "stage4.input.stage3.registers.Rs1Value",
        ),
        (
            "stage3.registers_claim_reduction.eval.Rs2Value",
            "stage4.input.stage3.registers.Rs2Value",
        ),
        (
            "stage3.instruction_input.eval.Rs1Value",
            "stage4.input.stage3.instruction.Rs1Value",
        ),
        (
            "stage3.instruction_input.eval.Rs2Value",
            "stage4.input.stage3.instruction.Rs2Value",
        ),
    ] {
        inputs.push(Stage4OpeningInputValue {
            symbol,
            point: stage3_point.clone(),
            eval: stage_eval_stage3(stage3, eval_name),
        });
    }
    inputs.push(Stage4OpeningInputValue {
        symbol: "stage4.input.stage2.RamVal",
        point: ram_val_point,
        eval: stage_eval_stage2(stage2, "stage2.ram_read_write.eval.RamVal"),
    });
    inputs.push(Stage4OpeningInputValue {
        symbol: "stage4.input.stage2.RamValFinal",
        point: ram_final_point.clone(),
        eval: stage_eval_stage2(stage2, "stage2.ram_output.eval.RamValFinal"),
    });
    inputs.push(Stage4OpeningInputValue {
        symbol: "stage4.input.initial_ram.RamValInit",
        point: ram_final_point.clone(),
        eval: mle_eval_u64(initial_ram_state, &ram_final_point),
    });
    inputs
}

fn stage5_opening_inputs(
    params: &JoltProtocolParams,
    stage2_inputs: &[Stage2OpeningInputValue<Fr>],
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
) -> Vec<generated_stage5::Stage5OpeningInputValue<Fr>> {
    let stage2 = stage2_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage2.sumcheck")
        .expect("Stage 2 batched output");
    let stage4 = stage4_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage4.sumcheck")
        .expect("Stage 4 batched output");

    let instruction_point = reversed_suffix(&stage2.point, params.log_t);
    let stage2_ram_rw_point = reverse_point(&stage2.point);
    let stage2_ram_raf_address = reversed_suffix(&stage2.point, params.log_k_ram);
    let stage2_ram_address_input = stage2_inputs
        .iter()
        .find(|input| input.symbol == "stage2.input.stage1.RamAddress")
        .expect("Stage 2 RamAddress input");
    let mut stage2_ram_raf_point = stage2_ram_raf_address;
    stage2_ram_raf_point.extend_from_slice(&stage2_ram_address_input.point);

    let stage4_ram_address = stage2_ram_rw_point[..params.log_k_ram].to_vec();
    let mut stage4_ram_val_check_point = stage4_ram_address;
    stage4_ram_val_check_point.extend(reversed_suffix(&stage4.point, params.log_t));
    let stage4_registers_point = normalized_stage4_registers_rw_point(params, &stage4.point);

    vec![
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.instruction.LookupOutput",
            point: instruction_point.clone(),
            eval: stage_eval_stage2(
                stage2,
                "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
            ),
        },
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.product_virtual.LookupOutput",
            point: instruction_point.clone(),
            eval: stage_eval_stage2(stage2, "stage2.product_virtual.remainder.eval.LookupOutput"),
        },
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.instruction.LeftLookupOperand",
            point: instruction_point.clone(),
            eval: stage_eval_stage2(
                stage2,
                "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
            ),
        },
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.instruction.RightLookupOperand",
            point: instruction_point,
            eval: stage_eval_stage2(
                stage2,
                "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            ),
        },
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.ram_raf.RamRa",
            point: stage2_ram_raf_point,
            eval: stage_eval_stage2(stage2, "stage2.ram_raf.eval.RamRa"),
        },
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.ram_read_write.RamRa",
            point: stage2_ram_rw_point,
            eval: stage_eval_stage2(stage2, "stage2.ram_read_write.eval.RamRa"),
        },
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage4.ram_val_check.RamRa",
            point: stage4_ram_val_check_point,
            eval: stage_eval_stage4(stage4, "stage4.ram_val_check.eval.RamRa"),
        },
        generated_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage4.registers.RegistersVal",
            point: stage4_registers_point,
            eval: stage_eval_stage4(stage4, "stage4.registers_read_write.eval.RegistersVal"),
        },
    ]
}

fn stage_eval_stage2(
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
    name: &'static str,
) -> Fr {
    output
        .evals
        .iter()
        .find(|eval| eval.name == name)
        .unwrap_or_else(|| panic!("missing eval {name}"))
        .value
}

fn stage_eval_stage3(
    output: &jolt_kernels::stage3::Stage3SumcheckOutput<Fr>,
    name: &'static str,
) -> Fr {
    output
        .evals
        .iter()
        .find(|eval| eval.name == name)
        .unwrap_or_else(|| panic!("missing eval {name}"))
        .value
}

fn stage_eval_stage4(
    output: &jolt_kernels::stage4::Stage4SumcheckOutput<Fr>,
    name: &'static str,
) -> Fr {
    output
        .evals
        .iter()
        .find(|eval| eval.name == name)
        .unwrap_or_else(|| panic!("missing eval {name}"))
        .value
}

fn reverse_point(point: &[Fr]) -> Vec<Fr> {
    point.iter().rev().copied().collect()
}

fn reversed_suffix(point: &[Fr], len: usize) -> Vec<Fr> {
    let start = point
        .len()
        .checked_sub(len)
        .unwrap_or_else(|| panic!("point is shorter than suffix length {len}"));
    point[start..].iter().rev().copied().collect()
}

fn normalized_stage4_registers_rw_point(params: &JoltProtocolParams, point: &[Fr]) -> Vec<Fr> {
    let expected = params.log_t + params.register_log_k;
    assert_eq!(
        point.len(),
        expected,
        "Stage 4 registers point length mismatch"
    );
    let (cycle, address) = point.split_at(params.log_t);
    address
        .iter()
        .rev()
        .copied()
        .chain(cycle.iter().rev().copied())
        .collect()
}

fn stage4_rd_inc(accesses: &[Stage4RegisterAccess]) -> Vec<Fr> {
    accesses
        .iter()
        .map(|access| {
            access.rd.map_or(Fr::from_u64(0), |rd| {
                field_delta_u64(rd.post_value, rd.pre_value)
            })
        })
        .collect()
}

fn stage4_rd_write_addresses(accesses: &[Stage4RegisterAccess]) -> Vec<Option<usize>> {
    accesses
        .iter()
        .map(|access| access.rd.map(|rd| rd.address))
        .collect()
}

fn stage4_ram_address_indices(accesses: &[Stage2RamAccess]) -> Vec<Option<usize>> {
    accesses
        .iter()
        .map(|access| access.remapped_address)
        .collect()
}

fn stage2_ram_inc(accesses: &[Stage2RamAccess]) -> Vec<Fr> {
    accesses
        .iter()
        .map(|access| field_delta_u64(access.write_value, access.read_value))
        .collect()
}

fn stage6_witness_polynomials(
    fixture: &CoreMuldivCommitmentFixture,
    opening_inputs: &[kernel_stage6::Stage6OpeningInputValue<Fr>],
) -> Stage6WitnessPolynomials {
    let one_hot_params = jolt_core::zkvm::config::OneHotParams::from_config(
        &fixture.proof.one_hot_config,
        fixture.verifier_preprocessing.shared.bytecode.code_size,
        fixture.proof.ram_K,
    );
    let bytecode = fixture.verifier_preprocessing.shared.bytecode.as_ref();
    let memory_layout = &fixture.verifier_preprocessing.shared.memory_layout;
    let trace = fixture.padded_trace.as_slice();
    let trace_len = fixture.proof.trace_length;
    assert_eq!(trace.len(), trace_len, "padded trace length mismatch");

    let instruction_indices = (0..fixture.params.instruction_d)
        .map(|index| instruction_ra_chunk_indices(trace, &one_hot_params, index))
        .collect::<Vec<_>>();
    let bytecode_indices = (0..fixture.params.bytecode_d)
        .map(|index| bytecode_ra_chunk_indices(trace, bytecode, &one_hot_params, index))
        .collect::<Vec<_>>();
    let ram_indices = (0..fixture.params.ram_d)
        .map(|index| ram_ra_chunk_indices(trace, memory_layout, &one_hot_params, index))
        .collect::<Vec<_>>();

    let instruction_ra_booleanity = instruction_indices
        .iter()
        .map(|indices| one_hot_cycle_major_from_indices(indices, fixture.params.log_k_chunk))
        .collect::<Vec<_>>();
    let bytecode_ra_booleanity = bytecode_indices
        .iter()
        .map(|indices| one_hot_cycle_major_from_indices(indices, fixture.params.log_k_chunk))
        .collect::<Vec<_>>();
    let ram_ra_booleanity = ram_indices
        .iter()
        .map(|indices| one_hot_cycle_major_from_indices(indices, fixture.params.log_k_chunk))
        .collect::<Vec<_>>();

    let bytecode_ra_read_raf = bytecode_indices
        .iter()
        .zip(stage6_bytecode_chunk_lens(&fixture.params))
        .map(|(indices, chunk_len)| one_hot_address_major_from_indices(indices, chunk_len))
        .collect::<Vec<_>>();

    let ram_address_chunks = stage6_ram_virtual_address_chunks(&fixture.params, opening_inputs);
    assert_eq!(
        ram_address_chunks.len(),
        fixture.params.ram_d,
        "RAM Stage 6 address chunk count mismatch"
    );
    let ram_ra_virtual = ram_indices
        .iter()
        .zip(&ram_address_chunks)
        .map(|(indices, point)| one_hot_evals_at_chunk_point(indices, point))
        .collect::<Vec<_>>();

    let instruction_address_chunks =
        stage6_instruction_virtual_address_chunks(&fixture.params, opening_inputs);
    assert_eq!(
        instruction_address_chunks.len(),
        fixture.params.instruction_d,
        "instruction Stage 6 address chunk count mismatch"
    );
    let instruction_ra_virtual = instruction_indices
        .iter()
        .zip(&instruction_address_chunks)
        .map(|(indices, point)| one_hot_evals_at_chunk_point(indices, point))
        .collect::<Vec<_>>();

    let hamming_weight = trace
        .iter()
        .map(|cycle| {
            if cycle.ram_access().address() != 0 {
                Fr::from_u64(1)
            } else {
                Fr::from_u64(0)
            }
        })
        .collect();

    Stage6WitnessPolynomials {
        instruction_ra_booleanity,
        bytecode_ra_booleanity,
        ram_ra_booleanity,
        bytecode_ra_read_raf,
        instruction_ra_virtual,
        ram_ra_virtual,
        hamming_weight,
        ram_inc: stage2_ram_inc(&fixture.ram_accesses),
        rd_inc: stage4_rd_inc(&fixture.stage4_register_accesses),
    }
}

fn instruction_ra_chunk_indices(
    trace: &[jolt_trace::Cycle],
    one_hot_params: &jolt_core::zkvm::config::OneHotParams,
    chunk: usize,
) -> Vec<Option<u8>> {
    trace
        .iter()
        .map(|cycle| {
            let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
            Some(one_hot_params.lookup_index_chunk(lookup_index, chunk))
        })
        .collect()
}

fn bytecode_ra_chunk_indices(
    trace: &[jolt_trace::Cycle],
    bytecode: &jolt_core::zkvm::bytecode::BytecodePreprocessing,
    one_hot_params: &jolt_core::zkvm::config::OneHotParams,
    chunk: usize,
) -> Vec<Option<u8>> {
    trace
        .iter()
        .map(|cycle| {
            let pc = bytecode.get_pc(cycle);
            Some(one_hot_params.bytecode_pc_chunk(pc, chunk))
        })
        .collect()
}

fn ram_ra_chunk_indices(
    trace: &[jolt_trace::Cycle],
    memory_layout: &common::jolt_device::MemoryLayout,
    one_hot_params: &jolt_core::zkvm::config::OneHotParams,
    chunk: usize,
) -> Vec<Option<u8>> {
    trace
        .iter()
        .map(|cycle| {
            remap_address(cycle.ram_access().address() as u64, memory_layout)
                .map(|address| one_hot_params.ram_address_chunk(address, chunk))
        })
        .collect()
}

fn one_hot_cycle_major_from_indices(indices: &[Option<u8>], chunk_len: usize) -> Vec<Fr> {
    let chunk_domain = 1usize << chunk_len;
    let mut output = vec![Fr::from_u64(0); chunk_domain * indices.len()];
    for (cycle, index) in indices.iter().enumerate() {
        if let Some(index) = index {
            let index = usize::from(*index);
            assert!(
                index < chunk_domain,
                "one-hot index {index} exceeds chunk domain {chunk_domain}"
            );
            output[cycle * chunk_domain + index] = Fr::from_u64(1);
        }
    }
    output
}

fn one_hot_address_major_from_indices(indices: &[Option<u8>], chunk_len: usize) -> Vec<Fr> {
    let chunk_domain = 1usize << chunk_len;
    let mut output = vec![Fr::from_u64(0); chunk_domain * indices.len()];
    for (cycle, index) in indices.iter().enumerate() {
        if let Some(index) = index {
            let index = usize::from(*index);
            assert!(
                index < chunk_domain,
                "one-hot index {index} exceeds chunk domain {chunk_domain}"
            );
            output[index * indices.len() + cycle] = Fr::from_u64(1);
        }
    }
    output
}

fn one_hot_evals_at_chunk_point(indices: &[Option<u8>], point: &[Fr]) -> Vec<Fr> {
    let eq_table = EqPolynomial::<Fr>::evals(point, None);
    indices
        .iter()
        .map(|index| {
            index.map_or(Fr::from_u64(0), |index| {
                eq_table
                    .get(usize::from(index))
                    .copied()
                    .expect("one-hot index is inside chunk point domain")
            })
        })
        .collect()
}

fn stage6_bytecode_chunk_lens(params: &JoltProtocolParams) -> Vec<usize> {
    let first_chunk_len = params.log_k_bytecode % params.log_k_chunk;
    let mut chunk_lens = Vec::with_capacity(params.bytecode_d);
    if first_chunk_len != 0 {
        chunk_lens.push(first_chunk_len);
    }
    while chunk_lens.len() < params.bytecode_d {
        chunk_lens.push(params.log_k_chunk);
    }
    chunk_lens
}

fn stage6_ram_virtual_address_chunks(
    params: &JoltProtocolParams,
    opening_inputs: &[kernel_stage6::Stage6OpeningInputValue<Fr>],
) -> Vec<Vec<Fr>> {
    let point = stage6_opening_point(
        opening_inputs,
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    );
    assert!(
        point.len() >= params.log_k_ram,
        "RAM RA opening point is shorter than the RAM address arity"
    );
    address_chunks_from_point(&point[..params.log_k_ram], params.log_k_chunk)
}

fn stage6_instruction_virtual_address_chunks(
    params: &JoltProtocolParams,
    opening_inputs: &[kernel_stage6::Stage6OpeningInputValue<Fr>],
) -> Vec<Vec<Fr>> {
    let mut address = Vec::with_capacity(params.instruction_log_k);
    for index in 0..params.instruction_ra_virtual_d {
        let symbol = format!("stage6.input.stage5.instruction_read_raf.InstructionRa_{index}");
        let point = stage6_opening_point(opening_inputs, &symbol);
        assert!(
            point.len() >= params.lookups_ra_virtual_log_k_chunk,
            "instruction RA opening point is shorter than the virtual address chunk arity"
        );
        address.extend_from_slice(&point[..params.lookups_ra_virtual_log_k_chunk]);
    }
    address_chunks_from_point(&address, params.log_k_chunk)
}

fn address_chunks_from_point(address: &[Fr], log_k_chunk: usize) -> Vec<Vec<Fr>> {
    let mut padded: Vec<Fr> = Vec::new();
    let remainder = address.len() % log_k_chunk;
    if remainder != 0 {
        padded.resize(log_k_chunk - remainder, Fr::from_u64(0));
    }
    padded.extend_from_slice(address);
    padded
        .chunks(log_k_chunk)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn stage6_opening_point<'a>(
    opening_inputs: &'a [kernel_stage6::Stage6OpeningInputValue<Fr>],
    symbol: &str,
) -> &'a [Fr] {
    opening_inputs
        .iter()
        .find(|input| input.symbol == symbol)
        .map(|input| input.point.as_slice())
        .unwrap_or_else(|| panic!("missing Stage 6 opening input `{symbol}`"))
}

fn field_delta_u64(post: u64, pre: u64) -> Fr {
    if post >= pre {
        Fr::from_u64(post - pre)
    } else {
        -Fr::from_u64(pre - post)
    }
}

fn mle_eval_u64(values: &[u64], point: &[Fr]) -> Fr {
    EqPolynomial::<Fr>::evals(point, None)
        .iter()
        .zip(values)
        .map(|(&weight, &value)| weight * Fr::from_u64(value))
        .sum()
}

fn core_stage4_artifacts(fixture: &CoreMuldivCommitmentFixture) -> CoreStage4Artifacts {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let stage4_result = verifier.verify_stage4().expect("core Stage 4 verifies");

    let round_polynomials = core_stage4_round_polynomials(
        &fixture.proof,
        stage4_result.initial_claim,
        &stage4_result.challenges,
    );
    let point = stage4_result
        .challenges
        .iter()
        .copied()
        .map(Fr::from)
        .collect::<Vec<_>>();
    let output = Stage4SumcheckOutput {
        driver: "stage4.sumcheck",
        point,
        evals: core_stage4_evals(&verifier.opening_accumulator),
        proof: jolt_sumcheck::SumcheckProof { round_polynomials },
    };

    CoreStage4Artifacts {
        artifacts: Stage4ExecutionArtifacts {
            challenge_vectors: Vec::new(),
            sumchecks: vec![output],
            opening_batches: Vec::new(),
        },
        opening_inputs: core_stage4_opening_inputs(
            &verifier.opening_accumulator,
            &fixture.initial_ram_state,
        ),
        transcript_states: verifier.transcript.state_history[1..].to_vec(),
    }
}

fn core_stage5_artifacts(fixture: &CoreMuldivCommitmentFixture) -> CoreStage5Artifacts {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let stage5_result = verifier.verify_stage5().expect("core Stage 5 verifies");

    let round_polynomials = core_stage5_round_polynomials(
        &fixture.proof,
        stage5_result.initial_claim,
        &stage5_result.challenges,
    );
    let point = stage5_result
        .challenges
        .iter()
        .copied()
        .map(Fr::from)
        .collect::<Vec<_>>();
    let output = generated_stage5::Stage5SumcheckOutput {
        driver: "stage5.sumcheck",
        point,
        evals: core_stage5_evals(&verifier.opening_accumulator, &fixture.params),
        proof: jolt_sumcheck::SumcheckProof { round_polynomials },
    };

    CoreStage5Artifacts {
        proof: generated_stage5::Stage5Proof {
            sumchecks: vec![output],
        },
        opening_inputs: core_stage5_opening_inputs(&verifier.opening_accumulator),
        transcript_states: verifier.transcript.state_history[1..].to_vec(),
    }
}

fn core_stage6_artifacts(fixture: &CoreMuldivCommitmentFixture) -> CoreStage6Artifacts {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let stage6_result = verifier.verify_stage6().expect("core Stage 6 verifies");

    let round_polynomials = core_stage6_round_polynomials(
        &fixture.proof,
        stage6_result.initial_claim,
        &stage6_result.challenges,
    );
    let point = stage6_result
        .challenges
        .iter()
        .copied()
        .map(Fr::from)
        .collect::<Vec<_>>();
    let output = generated_stage6::Stage6SumcheckOutput {
        driver: "stage6.sumcheck",
        point,
        evals: core_stage6_evals(&verifier.opening_accumulator, &fixture.params),
        proof: jolt_sumcheck::SumcheckProof { round_polynomials },
    };

    CoreStage6Artifacts {
        proof: generated_stage6::Stage6Proof {
            sumchecks: vec![output],
        },
        opening_inputs: core_stage6_opening_inputs(&verifier.opening_accumulator, &fixture.params),
        transcript_states: verifier.transcript.state_history[1..].to_vec(),
    }
}

fn core_stage7_artifacts(fixture: &CoreMuldivCommitmentFixture) -> CoreStage7Artifacts {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let _ = verifier.verify_stage6().expect("core Stage 6 verifies");
    let stage7_result = verifier.verify_stage7().expect("core Stage 7 verifies");

    let round_polynomials = core_stage7_round_polynomials(
        &fixture.proof,
        stage7_result.initial_claim,
        &stage7_result.challenges,
    );
    let point = stage7_result
        .challenges
        .iter()
        .copied()
        .map(Fr::from)
        .collect::<Vec<_>>();
    let output = generated_stage7::Stage7SumcheckOutput {
        driver: "stage7.sumcheck",
        point,
        evals: core_stage7_evals(&verifier.opening_accumulator, &fixture.params),
        proof: jolt_sumcheck::SumcheckProof { round_polynomials },
    };

    CoreStage7Artifacts {
        proof: generated_stage7::Stage7Proof {
            sumchecks: vec![output],
        },
        opening_inputs: core_stage7_opening_inputs(&verifier.opening_accumulator, &fixture.params),
        transcript_states: verifier.transcript.state_history[1..].to_vec(),
    }
}

fn core_stage8_transcript_states(fixture: &CoreMuldivCommitmentFixture) -> Vec<[u8; 32]> {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let _ = verifier.verify_stage6().expect("core Stage 6 verifies");
    let _ = verifier.verify_stage7().expect("core Stage 7 verifies");
    let _ = verifier.verify_stage8().expect("core Stage 8 verifies");
    verifier.transcript.state_history[1..].to_vec()
}

fn assert_core_accepts_bolt_evaluation_proof(
    fixture: &CoreMuldivCommitmentFixture,
    evaluation: &jolt_verifier::JoltEvaluationProof,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.joint_opening_proof = evaluation.joint_opening_proof.0.clone();

    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        proof,
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core accepts Stage 1");
    let _ = verifier.verify_stage2().expect("core accepts Stage 2");
    let _ = verifier.verify_stage3().expect("core accepts Stage 3");
    let _ = verifier.verify_stage4().expect("core accepts Stage 4");
    let _ = verifier.verify_stage5().expect("core accepts Stage 5");
    let _ = verifier.verify_stage6().expect("core accepts Stage 6");
    let _ = verifier.verify_stage7().expect("core accepts Stage 7");
    let _ = verifier
        .verify_stage8()
        .expect("core accepts Bolt evaluation proof");
}

fn assert_dory_proofs_match(expected: &DoryProof, actual: &DoryProof) {
    assert_eq!(
        dory_proof_bytes(expected),
        dory_proof_bytes(actual),
        "Dory joint opening proof mismatch"
    );
}

fn dory_proof_bytes(proof: &DoryProof) -> Vec<u8> {
    postcard::to_stdvec(proof).expect("serialize Dory proof")
}

fn unrelated_dory_proof() -> DoryProof {
    let prover_setup = DoryScheme::setup_prover(1);
    let poly = Polynomial::new(vec![Fr::from_u64(0), Fr::from_u64(1)]);
    let point = vec![Fr::from_u64(7)];
    let mut transcript = jolt_transcript::Blake2bTranscript::new(b"unrelated-dory-proof");
    DoryScheme::open(
        &poly,
        &point,
        point[0],
        &prover_setup,
        None,
        &mut transcript,
    )
}

fn assert_stage4_artifacts_match(
    expected: &Stage4ExecutionArtifacts<Fr>,
    actual: &Stage4ExecutionArtifacts<Fr>,
) {
    assert_eq!(
        expected.sumchecks.len(),
        actual.sumchecks.len(),
        "Stage 4 sumcheck count mismatch"
    );
    for (sumcheck_index, (expected, actual)) in
        expected.sumchecks.iter().zip(&actual.sumchecks).enumerate()
    {
        assert_eq!(expected.driver, actual.driver, "Stage 4 driver mismatch");
        assert_eq!(
            expected.point, actual.point,
            "Stage 4 point mismatch for sumcheck {sumcheck_index}"
        );
        assert_eq!(
            expected.evals.len(),
            actual.evals.len(),
            "Stage 4 eval count mismatch for sumcheck {sumcheck_index}"
        );
        for (eval_index, (expected, actual)) in expected.evals.iter().zip(&actual.evals).enumerate()
        {
            assert_eq!(
                expected.name, actual.name,
                "Stage 4 eval name mismatch at {sumcheck_index}.{eval_index}"
            );
            assert_eq!(
                expected.value, actual.value,
                "Stage 4 eval value mismatch at {sumcheck_index}.{eval_index}"
            );
        }
        assert_eq!(
            expected.proof.round_polynomials.len(),
            actual.proof.round_polynomials.len(),
            "Stage 4 round count mismatch for sumcheck {sumcheck_index}"
        );
        for (round, (expected, actual)) in expected
            .proof
            .round_polynomials
            .iter()
            .zip(&actual.proof.round_polynomials)
            .enumerate()
        {
            assert_eq!(
                expected.clone().into_coefficients(),
                actual.clone().into_coefficients(),
                "Stage 4 polynomial mismatch at {sumcheck_index}.{round}"
            );
        }
    }
}

fn assert_stage5_artifacts_match(
    expected: &generated_stage5::Stage5Proof<Fr>,
    actual: &generated_stage5::Stage5Proof<Fr>,
) {
    assert_eq!(
        expected.sumchecks.len(),
        actual.sumchecks.len(),
        "Stage 5 sumcheck count mismatch"
    );
    for (sumcheck_index, (expected, actual)) in
        expected.sumchecks.iter().zip(&actual.sumchecks).enumerate()
    {
        assert_eq!(expected.driver, actual.driver, "Stage 5 driver mismatch");
        assert_eq!(
            expected.point, actual.point,
            "Stage 5 point mismatch for sumcheck {sumcheck_index}"
        );
        assert_eq!(
            expected.evals.len(),
            actual.evals.len(),
            "Stage 5 eval count mismatch for sumcheck {sumcheck_index}"
        );
        for (eval_index, (expected, actual)) in expected.evals.iter().zip(&actual.evals).enumerate()
        {
            assert_eq!(
                expected.name, actual.name,
                "Stage 5 eval name mismatch at {sumcheck_index}.{eval_index}"
            );
            assert_eq!(
                expected.value, actual.value,
                "Stage 5 eval value mismatch at {sumcheck_index}.{eval_index}"
            );
        }
        assert_eq!(
            expected.proof.round_polynomials.len(),
            actual.proof.round_polynomials.len(),
            "Stage 5 round count mismatch for sumcheck {sumcheck_index}"
        );
        for (round, (expected, actual)) in expected
            .proof
            .round_polynomials
            .iter()
            .zip(&actual.proof.round_polynomials)
            .enumerate()
        {
            assert_eq!(
                expected.clone().into_coefficients(),
                actual.clone().into_coefficients(),
                "Stage 5 polynomial mismatch at {sumcheck_index}.{round}"
            );
        }
    }
}

fn assert_stage6_artifacts_match(
    expected: &generated_stage6::Stage6Proof<Fr>,
    actual: &generated_stage6::Stage6ExecutionArtifacts<Fr>,
) {
    assert_eq!(
        expected.sumchecks.len(),
        actual.sumchecks.len(),
        "Stage 6 sumcheck count mismatch"
    );
    for (sumcheck_index, (expected, actual)) in
        expected.sumchecks.iter().zip(&actual.sumchecks).enumerate()
    {
        assert_eq!(expected.driver, actual.driver, "Stage 6 driver mismatch");
        assert_eq!(
            expected.proof.round_polynomials.len(),
            actual.proof.round_polynomials.len(),
            "Stage 6 round count mismatch for sumcheck {sumcheck_index}"
        );
        for (round, (expected, actual)) in expected
            .proof
            .round_polynomials
            .iter()
            .zip(&actual.proof.round_polynomials)
            .enumerate()
        {
            assert_eq!(
                expected.clone().into_coefficients(),
                actual.clone().into_coefficients(),
                "Stage 6 polynomial mismatch at {sumcheck_index}.{round}"
            );
        }
        assert_eq!(
            expected.evals.len(),
            actual.evals.len(),
            "Stage 6 eval count mismatch for sumcheck {sumcheck_index}"
        );
        for (eval_index, (expected, actual)) in expected.evals.iter().zip(&actual.evals).enumerate()
        {
            assert_eq!(
                expected.name, actual.name,
                "Stage 6 eval name mismatch at {sumcheck_index}.{eval_index}"
            );
            assert_eq!(
                expected.value, actual.value,
                "Stage 6 eval value mismatch at {sumcheck_index}.{eval_index}"
            );
        }
        assert_eq!(
            expected.point, actual.point,
            "Stage 6 point mismatch for sumcheck {sumcheck_index}"
        );
    }
}

fn assert_stage7_artifacts_match(
    expected: &generated_stage7::Stage7Proof<Fr>,
    actual: &generated_stage7::Stage7ExecutionArtifacts<Fr>,
) {
    assert_eq!(
        expected.sumchecks.len(),
        actual.sumchecks.len(),
        "Stage 7 sumcheck count mismatch"
    );
    for (sumcheck_index, (expected, actual)) in
        expected.sumchecks.iter().zip(&actual.sumchecks).enumerate()
    {
        assert_eq!(expected.driver, actual.driver, "Stage 7 driver mismatch");
        assert_eq!(
            expected.proof.round_polynomials.len(),
            actual.proof.round_polynomials.len(),
            "Stage 7 round count mismatch for sumcheck {sumcheck_index}"
        );
        for (round, (expected, actual)) in expected
            .proof
            .round_polynomials
            .iter()
            .zip(&actual.proof.round_polynomials)
            .enumerate()
        {
            assert_eq!(
                expected.clone().into_coefficients(),
                actual.clone().into_coefficients(),
                "Stage 7 polynomial mismatch at {sumcheck_index}.{round}"
            );
        }
        assert_eq!(
            expected.evals.len(),
            actual.evals.len(),
            "Stage 7 eval count mismatch for sumcheck {sumcheck_index}"
        );
        for (eval_index, (expected, actual)) in expected.evals.iter().zip(&actual.evals).enumerate()
        {
            assert_eq!(
                expected.name, actual.name,
                "Stage 7 eval name mismatch at {sumcheck_index}.{eval_index}"
            );
            assert_eq!(
                expected.value, actual.value,
                "Stage 7 eval value mismatch at {sumcheck_index}.{eval_index}"
            );
        }
        assert_eq!(
            expected.point, actual.point,
            "Stage 7 point mismatch for sumcheck {sumcheck_index}"
        );
    }
}

fn assert_commitments_match(
    expected: &[Option<DoryCommitment>],
    actual: &[Option<DoryCommitment>],
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "commitment count mismatch: expected {}, got {}",
        expected.len(),
        actual.len()
    );
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.is_some(),
            actual.is_some(),
            "commitment presence mismatch at {index}"
        );
        assert!(
            expected == actual,
            "commitment value mismatch at index {index}"
        );
    }
}

fn assert_stage4_opening_inputs_match(
    expected: &[Stage4OpeningInputValue<Fr>],
    actual: &[Stage4OpeningInputValue<Fr>],
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "Stage 4 opening input count mismatch"
    );
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.symbol, actual.symbol,
            "Stage 4 opening input symbol mismatch at {index}"
        );
        assert_eq!(
            expected.point, actual.point,
            "Stage 4 opening input point mismatch for {}",
            expected.symbol
        );
        assert_eq!(
            expected.eval, actual.eval,
            "Stage 4 opening input eval mismatch for {}",
            expected.symbol
        );
    }
}

fn assert_stage5_opening_inputs_match(
    expected: &[generated_stage5::Stage5OpeningInputValue<Fr>],
    actual: &[generated_stage5::Stage5OpeningInputValue<Fr>],
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "Stage 5 opening input count mismatch"
    );
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.symbol, actual.symbol,
            "Stage 5 opening input symbol mismatch at {index}"
        );
        assert_eq!(
            expected.point, actual.point,
            "Stage 5 opening input point mismatch for {}",
            expected.symbol
        );
        assert_eq!(
            expected.eval, actual.eval,
            "Stage 5 opening input eval mismatch for {}",
            expected.symbol
        );
    }
}

fn core_stage4_round_polynomials(
    proof: &CoreProof,
    initial_claim: CoreFr,
    challenges: &[CoreFr],
) -> Vec<UnivariatePoly<Fr>> {
    let core_polys = match &proof.stage4_sumcheck_proof {
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(proof) => {
            proof.decompress_all_rounds(initial_claim, challenges)
        }
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Zk(_) => {
            panic!("standard Stage 4 proof expected")
        }
    };
    core_polys
        .into_iter()
        .map(|coeffs| UnivariatePoly::new(coeffs.into_iter().map(Fr::from).collect()))
        .collect()
}

fn core_stage5_round_polynomials(
    proof: &CoreProof,
    initial_claim: CoreFr,
    challenges: &[CoreFr],
) -> Vec<UnivariatePoly<Fr>> {
    let core_polys = match &proof.stage5_sumcheck_proof {
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(proof) => {
            proof.decompress_all_rounds(initial_claim, challenges)
        }
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Zk(_) => {
            panic!("standard Stage 5 proof expected")
        }
    };
    core_polys
        .into_iter()
        .map(|coeffs| UnivariatePoly::new(coeffs.into_iter().map(Fr::from).collect()))
        .collect()
}

fn core_stage6_round_polynomials(
    proof: &CoreProof,
    initial_claim: CoreFr,
    challenges: &[CoreFr],
) -> Vec<UnivariatePoly<Fr>> {
    let core_polys = match &proof.stage6_sumcheck_proof {
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(proof) => {
            proof.decompress_all_rounds(initial_claim, challenges)
        }
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Zk(_) => {
            panic!("standard Stage 6 proof expected")
        }
    };
    core_polys
        .into_iter()
        .map(|coeffs| UnivariatePoly::new(coeffs.into_iter().map(Fr::from).collect()))
        .collect()
}

fn core_stage7_round_polynomials(
    proof: &CoreProof,
    initial_claim: CoreFr,
    challenges: &[CoreFr],
) -> Vec<UnivariatePoly<Fr>> {
    let core_polys = match &proof.stage7_sumcheck_proof {
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(proof) => {
            proof.decompress_all_rounds(initial_claim, challenges)
        }
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Zk(_) => {
            panic!("standard Stage 7 proof expected")
        }
    };
    core_polys
        .into_iter()
        .map(|coeffs| UnivariatePoly::new(coeffs.into_iter().map(Fr::from).collect()))
        .collect()
}

fn core_stage4_opening_inputs<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    initial_ram_state: &[u64],
) -> Vec<Stage4OpeningInputValue<Fr>> {
    let mut inputs = vec![
        core_stage4_opening_input(
            accumulator,
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
            "stage4.input.stage3.registers.RdWriteValue",
        ),
        core_stage4_opening_input(
            accumulator,
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
            "stage4.input.stage3.registers.Rs1Value",
        ),
        core_stage4_opening_input(
            accumulator,
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
            "stage4.input.stage3.registers.Rs2Value",
        ),
        core_stage4_opening_input(
            accumulator,
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
            "stage4.input.stage3.instruction.Rs1Value",
        ),
        core_stage4_opening_input(
            accumulator,
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
            "stage4.input.stage3.instruction.Rs2Value",
        ),
        core_stage4_opening_input(
            accumulator,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            "stage4.input.stage2.RamVal",
        ),
        core_stage4_opening_input(
            accumulator,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            "stage4.input.stage2.RamValFinal",
        ),
    ];
    let ram_final_point = inputs
        .iter()
        .find(|input| input.symbol == "stage4.input.stage2.RamValFinal")
        .expect("Stage 4 RamValFinal input")
        .point
        .clone();
    inputs.push(Stage4OpeningInputValue {
        symbol: "stage4.input.initial_ram.RamValInit",
        eval: mle_eval_u64(initial_ram_state, &ram_final_point),
        point: ram_final_point,
    });
    inputs
}

fn core_stage4_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> Stage4OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    Stage4OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage5_opening_inputs<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
) -> Vec<generated_stage5::Stage5OpeningInputValue<Fr>> {
    vec![
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            "stage5.input.stage2.instruction.LookupOutput",
        ),
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
            "stage5.input.stage2.product_virtual.LookupOutput",
        ),
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
            "stage5.input.stage2.instruction.LeftLookupOperand",
        ),
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
            "stage5.input.stage2.instruction.RightLookupOperand",
        ),
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            "stage5.input.stage2.ram_raf.RamRa",
        ),
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            "stage5.input.stage2.ram_read_write.RamRa",
        ),
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
            "stage5.input.stage4.ram_val_check.RamRa",
        ),
        core_stage5_virtual_opening_input(
            accumulator,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            "stage5.input.stage4.registers.RegistersVal",
        ),
    ]
}

fn core_stage5_virtual_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> generated_stage5::Stage5OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    generated_stage5::Stage5OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage6_opening_inputs<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    params: &JoltProtocolParams,
) -> Vec<generated_stage6::Stage6OpeningInputValue<Fr>> {
    let mut inputs = vec![
        core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
            "stage6.input.stage1.UnexpandedPC",
        ),
        core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::Imm,
            SumcheckId::SpartanOuter,
            "stage6.input.stage1.Imm",
        ),
    ];

    for (oracle, flag) in [
        ("AddOperands", CircuitFlags::AddOperands),
        ("SubtractOperands", CircuitFlags::SubtractOperands),
        ("MultiplyOperands", CircuitFlags::MultiplyOperands),
        ("Load", CircuitFlags::Load),
        ("Store", CircuitFlags::Store),
        ("Jump", CircuitFlags::Jump),
        ("WriteLookupOutputToRD", CircuitFlags::WriteLookupOutputToRD),
        ("VirtualInstruction", CircuitFlags::VirtualInstruction),
        ("Assert", CircuitFlags::Assert),
        (
            "DoNotUpdateUnexpandedPC",
            CircuitFlags::DoNotUpdateUnexpandedPC,
        ),
        ("Advice", CircuitFlags::Advice),
        ("IsCompressed", CircuitFlags::IsCompressed),
        ("IsFirstInSequence", CircuitFlags::IsFirstInSequence),
        ("IsLastInSequence", CircuitFlags::IsLastInSequence),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::OpFlags(flag),
            SumcheckId::SpartanOuter,
            leak_str(&format!("stage6.input.stage1.OpFlag{oracle}")),
        ));
    }

    for (symbol, polynomial) in [
        (
            "stage6.input.stage2.OpFlagJump",
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
        ),
        (
            "stage6.input.stage2.InstructionFlagBranch",
            VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
        ),
        (
            "stage6.input.stage2.OpFlagWriteLookupOutputToRD",
            VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
        ),
        (
            "stage6.input.stage2.OpFlagVirtualInstruction",
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
        ),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            polynomial,
            SumcheckId::SpartanProductVirtualization,
            symbol,
        ));
    }

    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::Imm,
        SumcheckId::InstructionInputVirtualization,
        "stage6.input.stage3.instruction_input.Imm",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::UnexpandedPC,
        SumcheckId::SpartanShift,
        "stage6.input.stage3.spartan_shift.UnexpandedPC",
    ));
    for (symbol, flag) in [
        (
            "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value",
            InstructionFlags::LeftOperandIsRs1Value,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC",
            InstructionFlags::LeftOperandIsPC,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value",
            InstructionFlags::RightOperandIsRs2Value,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm",
            InstructionFlags::RightOperandIsImm,
        ),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::InstructionFlags(flag),
            SumcheckId::InstructionInputVirtualization,
            symbol,
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
        SumcheckId::SpartanShift,
        "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop",
    ));
    for (symbol, flag) in [
        (
            "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction",
            CircuitFlags::VirtualInstruction,
        ),
        (
            "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence",
            CircuitFlags::IsFirstInSequence,
        ),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::OpFlags(flag),
            SumcheckId::SpartanShift,
            symbol,
        ));
    }

    for (symbol, polynomial) in [
        ("stage6.input.stage4.RdWa", VirtualPolynomial::RdWa),
        ("stage6.input.stage4.Rs1Ra", VirtualPolynomial::Rs1Ra),
        ("stage6.input.stage4.Rs2Ra", VirtualPolynomial::Rs2Ra),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            polynomial,
            SumcheckId::RegistersReadWriteChecking,
            symbol,
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::RdWa,
        SumcheckId::RegistersValEvaluation,
        "stage6.input.stage5.registers_val_evaluation.RdWa",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::InstructionRafFlag,
        SumcheckId::InstructionReadRaf,
        "stage6.input.stage5.InstructionRafFlag",
    ));
    for index in 0..params.lookup_table_count {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::LookupTableFlag(index),
            SumcheckId::InstructionReadRaf,
            leak_str(&format!("stage6.input.stage5.LookupTableFlag_{index}")),
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::PC,
        SumcheckId::SpartanOuter,
        "stage6.input.stage1.PC",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::PC,
        SumcheckId::SpartanShift,
        "stage6.input.stage3.spartan_shift.PC",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::RamRa,
        SumcheckId::RamRaClaimReduction,
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    ));
    for index in 0..params.instruction_ra_virtual_d {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::InstructionRa(index),
            SumcheckId::InstructionReadRaf,
            leak_str(&format!(
                "stage6.input.stage5.instruction_read_raf.InstructionRa_{index}"
            )),
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::LookupOutput,
        SumcheckId::SpartanOuter,
        "stage6.input.stage1.LookupOutput",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RamInc,
        SumcheckId::RamReadWriteChecking,
        "stage6.input.stage2.ram_read_write.RamInc",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RamInc,
        SumcheckId::RamValCheck,
        "stage6.input.stage4.ram_val_check.RamInc",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersReadWriteChecking,
        "stage6.input.stage4.registers_read_write.RdInc",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersValEvaluation,
        "stage6.input.stage5.registers_val_evaluation.RdInc",
    ));
    inputs
}

fn core_stage6_virtual_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> generated_stage6::Stage6OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    generated_stage6::Stage6OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage6_committed_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> generated_stage6::Stage6OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_committed_polynomial_opening(polynomial, sumcheck);
    generated_stage6::Stage6OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage7_opening_inputs<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    params: &JoltProtocolParams,
) -> Vec<generated_stage7::Stage7OpeningInputValue<Fr>> {
    let mut inputs = Vec::new();
    inputs.push(core_stage7_virtual_opening_input(
        accumulator,
        VirtualPolynomial::RamHammingWeight,
        SumcheckId::RamHammingBooleanity,
        "stage7.input.stage6.hamming_booleanity.HammingWeight",
    ));
    for index in 0..params.instruction_d {
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::InstructionRa(index),
            SumcheckId::Booleanity,
            leak_str(&format!(
                "stage7.input.stage6.booleanity.InstructionRa_{index}"
            )),
        ));
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::InstructionRa(index),
            SumcheckId::InstructionRaVirtualization,
            leak_str(&format!(
                "stage7.input.stage6.instruction_ra_virtual.InstructionRa_{index}"
            )),
        ));
    }
    for index in 0..params.bytecode_d {
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::BytecodeRa(index),
            SumcheckId::Booleanity,
            leak_str(&format!(
                "stage7.input.stage6.booleanity.BytecodeRa_{index}"
            )),
        ));
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::BytecodeRa(index),
            SumcheckId::BytecodeReadRaf,
            leak_str(&format!(
                "stage7.input.stage6.bytecode_read_raf.BytecodeRa_{index}"
            )),
        ));
    }
    for index in 0..params.ram_d {
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::RamRa(index),
            SumcheckId::Booleanity,
            leak_str(&format!("stage7.input.stage6.booleanity.RamRa_{index}")),
        ));
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::RamRa(index),
            SumcheckId::RamRaVirtualization,
            leak_str(&format!("stage7.input.stage6.ram_ra_virtual.RamRa_{index}")),
        ));
    }
    inputs
}

fn core_stage7_virtual_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> generated_stage7::Stage7OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    generated_stage7::Stage7OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage7_committed_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> generated_stage7::Stage7OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_committed_polynomial_opening(polynomial, sumcheck);
    generated_stage7::Stage7OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage5_evals<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    params: &JoltProtocolParams,
) -> Vec<generated_stage5::Stage5NamedEval<Fr>> {
    let mut evals = Vec::new();
    for index in 0..params.lookup_table_count {
        evals.push(core_stage5_virtual_eval(
            accumulator,
            Box::leak(
                format!("stage5.instruction_read_raf.eval.LookupTableFlag_{index}")
                    .into_boxed_str(),
            ),
            Box::leak(format!("LookupTableFlag_{index}").into_boxed_str()),
            VirtualPolynomial::LookupTableFlag(index),
            SumcheckId::InstructionReadRaf,
        ));
    }
    for index in 0..params.instruction_ra_virtual_d {
        evals.push(core_stage5_virtual_eval(
            accumulator,
            Box::leak(
                format!("stage5.instruction_read_raf.eval.InstructionRa_{index}").into_boxed_str(),
            ),
            Box::leak(format!("InstructionRa_{index}").into_boxed_str()),
            VirtualPolynomial::InstructionRa(index),
            SumcheckId::InstructionReadRaf,
        ));
    }
    evals.push(core_stage5_virtual_eval(
        accumulator,
        "stage5.instruction_read_raf.eval.InstructionRafFlag",
        "InstructionRafFlag",
        VirtualPolynomial::InstructionRafFlag,
        SumcheckId::InstructionReadRaf,
    ));
    evals.push(core_stage5_virtual_eval(
        accumulator,
        "stage5.ram_ra_claim_reduction.eval.RamRa",
        "RamRa",
        VirtualPolynomial::RamRa,
        SumcheckId::RamRaClaimReduction,
    ));
    evals.push(core_stage5_committed_eval(
        accumulator,
        "stage5.registers_val_evaluation.eval.RdInc",
        "RdInc",
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersValEvaluation,
    ));
    evals.push(core_stage5_virtual_eval(
        accumulator,
        "stage5.registers_val_evaluation.eval.RdWa",
        "RdWa",
        VirtualPolynomial::RdWa,
        SumcheckId::RegistersValEvaluation,
    ));
    evals
}

fn core_stage5_virtual_eval<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    name: &'static str,
    oracle: &'static str,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
) -> generated_stage5::Stage5NamedEval<Fr> {
    let (_, value) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    generated_stage5::Stage5NamedEval {
        name,
        oracle,
        value: Fr::from(value),
    }
}

fn core_stage5_committed_eval<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    name: &'static str,
    oracle: &'static str,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
) -> generated_stage5::Stage5NamedEval<Fr> {
    let (_, value) = accumulator.get_committed_polynomial_opening(polynomial, sumcheck);
    generated_stage5::Stage5NamedEval {
        name,
        oracle,
        value: Fr::from(value),
    }
}

fn core_stage6_evals<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    params: &JoltProtocolParams,
) -> Vec<generated_stage6::Stage6NamedEval<Fr>> {
    let mut evals = Vec::new();
    for index in 0..params.bytecode_d {
        evals.push(core_stage6_committed_eval(
            accumulator,
            leak_str(&format!("stage6.bytecode_read_raf.eval.BytecodeRa_{index}")),
            leak_str(&format!("BytecodeRa_{index}")),
            CommittedPolynomial::BytecodeRa(index),
            SumcheckId::BytecodeReadRaf,
        ));
    }
    for index in 0..params.instruction_d {
        evals.push(core_stage6_committed_eval(
            accumulator,
            leak_str(&format!("stage6.booleanity.eval.InstructionRa_{index}")),
            leak_str(&format!("InstructionRa_{index}")),
            CommittedPolynomial::InstructionRa(index),
            SumcheckId::Booleanity,
        ));
    }
    for index in 0..params.bytecode_d {
        evals.push(core_stage6_committed_eval(
            accumulator,
            leak_str(&format!("stage6.booleanity.eval.BytecodeRa_{index}")),
            leak_str(&format!("BytecodeRa_{index}")),
            CommittedPolynomial::BytecodeRa(index),
            SumcheckId::Booleanity,
        ));
    }
    for index in 0..params.ram_d {
        evals.push(core_stage6_committed_eval(
            accumulator,
            leak_str(&format!("stage6.booleanity.eval.RamRa_{index}")),
            leak_str(&format!("RamRa_{index}")),
            CommittedPolynomial::RamRa(index),
            SumcheckId::Booleanity,
        ));
    }
    evals.push(core_stage6_virtual_eval(
        accumulator,
        "stage6.hamming_booleanity.eval.HammingWeight",
        "HammingWeight",
        VirtualPolynomial::RamHammingWeight,
        SumcheckId::RamHammingBooleanity,
    ));
    for index in 0..params.ram_d {
        evals.push(core_stage6_committed_eval(
            accumulator,
            leak_str(&format!("stage6.ram_ra_virtual.eval.RamRa_{index}")),
            leak_str(&format!("RamRa_{index}")),
            CommittedPolynomial::RamRa(index),
            SumcheckId::RamRaVirtualization,
        ));
    }
    for index in 0..params.instruction_d {
        evals.push(core_stage6_committed_eval(
            accumulator,
            leak_str(&format!(
                "stage6.instruction_ra_virtual.eval.InstructionRa_{index}"
            )),
            leak_str(&format!("InstructionRa_{index}")),
            CommittedPolynomial::InstructionRa(index),
            SumcheckId::InstructionRaVirtualization,
        ));
    }
    evals.push(core_stage6_committed_eval(
        accumulator,
        "stage6.inc_claim_reduction.eval.RamInc",
        "RamInc",
        CommittedPolynomial::RamInc,
        SumcheckId::IncClaimReduction,
    ));
    evals.push(core_stage6_committed_eval(
        accumulator,
        "stage6.inc_claim_reduction.eval.RdInc",
        "RdInc",
        CommittedPolynomial::RdInc,
        SumcheckId::IncClaimReduction,
    ));
    evals
}

fn core_stage6_virtual_eval<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    name: &'static str,
    oracle: &'static str,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
) -> generated_stage6::Stage6NamedEval<Fr> {
    let (_, value) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    generated_stage6::Stage6NamedEval {
        name,
        oracle,
        value: Fr::from(value),
    }
}

fn core_stage6_committed_eval<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    name: &'static str,
    oracle: &'static str,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
) -> generated_stage6::Stage6NamedEval<Fr> {
    let (_, value) = accumulator.get_committed_polynomial_opening(polynomial, sumcheck);
    generated_stage6::Stage6NamedEval {
        name,
        oracle,
        value: Fr::from(value),
    }
}

fn core_stage7_evals<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    params: &JoltProtocolParams,
) -> Vec<generated_stage7::Stage7NamedEval<Fr>> {
    let mut evals = Vec::new();
    for index in 0..params.instruction_d {
        evals.push(core_stage7_committed_eval(
            accumulator,
            leak_str(&format!(
                "stage7.hamming_weight_claim_reduction.eval.InstructionRa_{index}"
            )),
            leak_str(&format!("InstructionRa_{index}")),
            CommittedPolynomial::InstructionRa(index),
            SumcheckId::HammingWeightClaimReduction,
        ));
    }
    for index in 0..params.bytecode_d {
        evals.push(core_stage7_committed_eval(
            accumulator,
            leak_str(&format!(
                "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_{index}"
            )),
            leak_str(&format!("BytecodeRa_{index}")),
            CommittedPolynomial::BytecodeRa(index),
            SumcheckId::HammingWeightClaimReduction,
        ));
    }
    for index in 0..params.ram_d {
        evals.push(core_stage7_committed_eval(
            accumulator,
            leak_str(&format!(
                "stage7.hamming_weight_claim_reduction.eval.RamRa_{index}"
            )),
            leak_str(&format!("RamRa_{index}")),
            CommittedPolynomial::RamRa(index),
            SumcheckId::HammingWeightClaimReduction,
        ));
    }
    evals
}

fn core_stage7_committed_eval<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    name: &'static str,
    oracle: &'static str,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
) -> generated_stage7::Stage7NamedEval<Fr> {
    let (_, value) = accumulator.get_committed_polynomial_opening(polynomial, sumcheck);
    generated_stage7::Stage7NamedEval {
        name,
        oracle,
        value: Fr::from(value),
    }
}

fn core_stage4_evals<A: OpeningAccumulator<CoreFr>>(accumulator: &A) -> Vec<Stage4NamedEval<Fr>> {
    let mut evals = [
        (
            "stage4.registers_read_write.eval.RegistersVal",
            "RegistersVal",
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        ),
        (
            "stage4.registers_read_write.eval.Rs1Ra",
            "Rs1Ra",
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        (
            "stage4.registers_read_write.eval.Rs2Ra",
            "Rs2Ra",
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        ),
        (
            "stage4.registers_read_write.eval.RdWa",
            "RdWa",
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        ),
    ]
    .into_iter()
    .map(|(name, oracle, polynomial, sumcheck)| {
        let (_, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
        Stage4NamedEval {
            name,
            oracle,
            value: Fr::from(eval),
        }
    })
    .collect::<Vec<_>>();
    let (_, rd_inc) = accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersReadWriteChecking,
    );
    evals.push(Stage4NamedEval {
        name: "stage4.registers_read_write.eval.RdInc",
        oracle: "RdInc",
        value: Fr::from(rd_inc),
    });
    evals.extend(
        [(
            "stage4.ram_val_check.eval.RamRa",
            "RamRa",
            VirtualPolynomial::RamRa,
            SumcheckId::RamValCheck,
        )]
        .into_iter()
        .map(|(name, oracle, polynomial, sumcheck)| {
            let (_, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
            Stage4NamedEval {
                name,
                oracle,
                value: Fr::from(eval),
            }
        }),
    );
    let (_, ram_inc) = accumulator
        .get_committed_polynomial_opening(CommittedPolynomial::RamInc, SumcheckId::RamValCheck);
    evals.push(Stage4NamedEval {
        name: "stage4.ram_val_check.eval.RamInc",
        oracle: "RamInc",
        value: Fr::from(ram_inc),
    });
    evals
}

fn generated_stage3_opening_inputs(
    inputs: &[Stage3OpeningInputValue<Fr>],
) -> Vec<generated_stage3::Stage3OpeningInputValue<Fr>> {
    inputs
        .iter()
        .map(|input| generated_stage3::Stage3OpeningInputValue {
            symbol: input.symbol,
            point: input.point.clone(),
            eval: input.eval,
        })
        .collect()
}

fn generated_stage4_opening_inputs(
    inputs: &[Stage4OpeningInputValue<Fr>],
) -> Vec<generated_stage4::Stage4OpeningInputValue<Fr>> {
    inputs
        .iter()
        .map(|input| generated_stage4::Stage4OpeningInputValue {
            symbol: input.symbol,
            point: input.point.clone(),
            eval: input.eval,
        })
        .collect()
}

fn kernel_stage5_opening_inputs(
    inputs: &[generated_stage5::Stage5OpeningInputValue<Fr>],
) -> Vec<kernel_stage5::Stage5OpeningInputValue<Fr>> {
    inputs
        .iter()
        .map(|input| kernel_stage5::Stage5OpeningInputValue {
            symbol: input.symbol,
            point: input.point.clone(),
            eval: input.eval,
        })
        .collect()
}

fn kernel_stage6_opening_inputs(
    inputs: &[generated_stage6::Stage6OpeningInputValue<Fr>],
) -> Vec<kernel_stage6::Stage6OpeningInputValue<Fr>> {
    inputs
        .iter()
        .map(|input| kernel_stage6::Stage6OpeningInputValue {
            symbol: input.symbol,
            point: input.point.clone(),
            eval: input.eval,
        })
        .collect()
}

fn kernel_stage7_opening_inputs(
    inputs: &[generated_stage7::Stage7OpeningInputValue<Fr>],
) -> Vec<jolt_kernels::stage7::Stage7OpeningInputValue<Fr>> {
    inputs
        .iter()
        .map(|input| jolt_kernels::stage7::Stage7OpeningInputValue {
            symbol: input.symbol,
            point: input.point.clone(),
            eval: input.eval,
        })
        .collect()
}

fn kernel_stage6_bytecode_entries(
    entries: &[generated_stage6::Stage6BytecodeEntry],
) -> Vec<kernel_stage6::Stage6BytecodeEntry<Fr>> {
    entries
        .iter()
        .map(|entry| kernel_stage6::Stage6BytecodeEntry {
            address: entry.address,
            imm: entry.imm,
            circuit_flags: entry.circuit_flags,
            rd: entry.rd,
            rs1: entry.rs1,
            rs2: entry.rs2,
            lookup_table: entry.lookup_table,
            is_interleaved: entry.is_interleaved,
            is_branch: entry.is_branch,
            left_is_rs1: entry.left_is_rs1,
            left_is_pc: entry.left_is_pc,
            right_is_rs2: entry.right_is_rs2,
            right_is_imm: entry.right_is_imm,
            is_noop: entry.is_noop,
        })
        .collect()
}

fn generated_stage2_opening_inputs(
    inputs: &[Stage2OpeningInputValue<Fr>],
) -> Vec<generated_stage2::Stage2OpeningInputValue<Fr>> {
    inputs
        .iter()
        .map(|input| generated_stage2::Stage2OpeningInputValue {
            symbol: input.symbol,
            point: input.point.clone(),
            eval: input.eval,
        })
        .collect()
}

fn generated_stage2_ram_accesses(
    accesses: &[Stage2RamAccess],
) -> Vec<generated_stage2::Stage2RamAccess> {
    accesses
        .iter()
        .map(|access| generated_stage2::Stage2RamAccess {
            remapped_address: access.remapped_address,
            read_value: access.read_value,
            write_value: access.write_value,
        })
        .collect()
}

fn generated_stage2_ram_output_layout(
    layout: Stage2RamOutputLayout,
) -> generated_stage2::Stage2RamOutputLayout {
    generated_stage2::Stage2RamOutputLayout {
        io_start: layout.io_start,
        io_end: layout.io_end,
    }
}

fn to_generated_stage1_proof(proof: &Stage1Proof<Fr>) -> generated_stage1::Stage1Proof<Fr> {
    generated_stage1::Stage1Proof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(to_generated_stage1_sumcheck_output)
            .collect(),
    }
}

fn to_generated_stage1_sumcheck_output(
    output: &jolt_kernels::stage1::Stage1SumcheckOutput<Fr>,
) -> generated_stage1::Stage1SumcheckOutput<Fr> {
    generated_stage1::Stage1SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| generated_stage1::Stage1NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_generated_stage2_proof(
    artifacts: &Stage2ExecutionArtifacts<Fr>,
) -> generated_stage2::Stage2Proof<Fr> {
    generated_stage2::Stage2Proof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_generated_stage2_sumcheck_output)
            .collect(),
    }
}

fn to_generated_stage2_sumcheck_output(
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) -> generated_stage2::Stage2SumcheckOutput<Fr> {
    generated_stage2::Stage2SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| generated_stage2::Stage2NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_generated_jolt_proof(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &generated_stage5::Stage5Proof<Fr>,
) -> jolt_verifier::JoltProof {
    jolt_verifier::JoltProof {
        commitments: commitments.to_vec(),
        stage1_outer: to_jolt_stage1_proof(stage1_artifacts),
        stage2: to_jolt_stage2_proof(stage2_artifacts),
        stage3: to_jolt_stage3_proof(stage3_artifacts),
        stage4: to_jolt_stage4_proof(stage4_artifacts),
        stage5: to_jolt_stage5_proof(stage5_proof),
        stage6: jolt_verifier::JoltStageProof::default(),
        stage7: jolt_verifier::JoltStageProof::default(),
        evaluation: None,
    }
}

fn to_generated_jolt_proof_with_stage6(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &generated_stage5::Stage5Proof<Fr>,
    stage6_proof: &generated_stage6::Stage6Proof<Fr>,
) -> jolt_verifier::JoltProof {
    let mut proof = to_generated_jolt_proof(
        commitments,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
    );
    proof.stage6 = to_jolt_stage6_proof(stage6_proof);
    proof
}

fn to_generated_jolt_proof_with_stage7(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &generated_stage5::Stage5Proof<Fr>,
    stage6_proof: &generated_stage6::Stage6Proof<Fr>,
    stage7_proof: &generated_stage7::Stage7Proof<Fr>,
) -> jolt_verifier::JoltProof {
    let mut proof = to_generated_jolt_proof_with_stage6(
        commitments,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
        stage6_proof,
    );
    proof.stage7 = to_jolt_stage7_proof(stage7_proof);
    proof
}

fn to_jolt_stage1_proof(artifacts: &Stage1ExecutionArtifacts<Fr>) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_jolt_stage1_sumcheck_output)
            .collect(),
    }
}

fn to_jolt_stage2_proof(artifacts: &Stage2ExecutionArtifacts<Fr>) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_jolt_stage2_sumcheck_output)
            .collect(),
    }
}

fn to_jolt_stage3_proof(artifacts: &Stage3ExecutionArtifacts<Fr>) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_jolt_stage3_sumcheck_output)
            .collect(),
    }
}

fn to_jolt_stage4_proof(artifacts: &Stage4ExecutionArtifacts<Fr>) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_jolt_stage4_sumcheck_output)
            .collect(),
    }
}

fn to_jolt_stage5_proof(
    proof: &generated_stage5::Stage5Proof<Fr>,
) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(to_jolt_stage5_sumcheck_output)
            .collect(),
    }
}

fn to_jolt_stage6_proof(
    proof: &generated_stage6::Stage6Proof<Fr>,
) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(to_jolt_stage6_sumcheck_output)
            .collect(),
    }
}

fn to_jolt_stage7_proof(
    proof: &generated_stage7::Stage7Proof<Fr>,
) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(to_jolt_stage7_sumcheck_output)
            .collect(),
    }
}

fn to_generated_stage5_proof(
    artifacts: &kernel_stage5::Stage5ExecutionArtifacts<Fr>,
) -> generated_stage5::Stage5Proof<Fr> {
    generated_stage5::Stage5Proof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_generated_stage5_sumcheck_output)
            .collect(),
    }
}

fn to_kernel_stage6_proof(
    proof: &generated_stage6::Stage6Proof<Fr>,
) -> kernel_stage6::Stage6Proof<Fr> {
    kernel_stage6::Stage6Proof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(to_kernel_stage6_sumcheck_output)
            .collect(),
    }
}

fn to_kernel_stage7_proof(
    proof: &generated_stage7::Stage7Proof<Fr>,
) -> jolt_kernels::stage7::Stage7Proof<Fr> {
    jolt_kernels::stage7::Stage7Proof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(to_kernel_stage7_sumcheck_output)
            .collect(),
    }
}

fn generated_stage6_execution_artifacts(
    artifacts: &kernel_stage6::Stage6ExecutionArtifacts<Fr>,
) -> generated_stage6::Stage6ExecutionArtifacts<Fr> {
    generated_stage6::Stage6ExecutionArtifacts {
        challenge_vectors: artifacts
            .challenge_vectors
            .iter()
            .map(|challenge| generated_stage6::Stage6ChallengeVector {
                symbol: challenge.symbol,
                values: challenge.values.clone(),
            })
            .collect(),
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_generated_stage6_sumcheck_output)
            .collect(),
        opening_batches: Vec::new(),
    }
}

fn generated_stage7_execution_artifacts(
    artifacts: &jolt_kernels::stage7::Stage7ExecutionArtifacts<Fr>,
) -> generated_stage7::Stage7ExecutionArtifacts<Fr> {
    generated_stage7::Stage7ExecutionArtifacts {
        challenge_vectors: artifacts
            .challenge_vectors
            .iter()
            .map(|challenge| generated_stage7::Stage7ChallengeVector {
                symbol: challenge.symbol,
                values: challenge.values.clone(),
            })
            .collect(),
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_generated_stage7_sumcheck_output)
            .collect(),
        opening_batches: Vec::new(),
    }
}

fn to_jolt_stage1_sumcheck_output(
    output: &jolt_kernels::stage1::Stage1SumcheckOutput<Fr>,
) -> jolt_verifier::JoltSumcheckOutput {
    jolt_verifier::JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_verifier::JoltNamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_jolt_stage2_sumcheck_output(
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) -> jolt_verifier::JoltSumcheckOutput {
    jolt_verifier::JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_verifier::JoltNamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_jolt_stage3_sumcheck_output(
    output: &jolt_kernels::stage3::Stage3SumcheckOutput<Fr>,
) -> jolt_verifier::JoltSumcheckOutput {
    jolt_verifier::JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_verifier::JoltNamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_jolt_stage4_sumcheck_output(
    output: &jolt_kernels::stage4::Stage4SumcheckOutput<Fr>,
) -> jolt_verifier::JoltSumcheckOutput {
    jolt_verifier::JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_verifier::JoltNamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_jolt_stage5_sumcheck_output(
    output: &generated_stage5::Stage5SumcheckOutput<Fr>,
) -> jolt_verifier::JoltSumcheckOutput {
    jolt_verifier::JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_verifier::JoltNamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_jolt_stage6_sumcheck_output(
    output: &generated_stage6::Stage6SumcheckOutput<Fr>,
) -> jolt_verifier::JoltSumcheckOutput {
    jolt_verifier::JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_verifier::JoltNamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_jolt_stage7_sumcheck_output(
    output: &generated_stage7::Stage7SumcheckOutput<Fr>,
) -> jolt_verifier::JoltSumcheckOutput {
    jolt_verifier::JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_verifier::JoltNamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_generated_stage5_sumcheck_output(
    output: &kernel_stage5::Stage5SumcheckOutput<Fr>,
) -> generated_stage5::Stage5SumcheckOutput<Fr> {
    generated_stage5::Stage5SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| generated_stage5::Stage5NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_kernel_stage6_sumcheck_output(
    output: &generated_stage6::Stage6SumcheckOutput<Fr>,
) -> kernel_stage6::Stage6SumcheckOutput<Fr> {
    kernel_stage6::Stage6SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| kernel_stage6::Stage6NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_kernel_stage7_sumcheck_output(
    output: &generated_stage7::Stage7SumcheckOutput<Fr>,
) -> jolt_kernels::stage7::Stage7SumcheckOutput<Fr> {
    jolt_kernels::stage7::Stage7SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| jolt_kernels::stage7::Stage7NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_generated_stage6_sumcheck_output(
    output: &kernel_stage6::Stage6SumcheckOutput<Fr>,
) -> generated_stage6::Stage6SumcheckOutput<Fr> {
    generated_stage6::Stage6SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| generated_stage6::Stage6NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_generated_stage7_sumcheck_output(
    output: &jolt_kernels::stage7::Stage7SumcheckOutput<Fr>,
) -> generated_stage7::Stage7SumcheckOutput<Fr> {
    generated_stage7::Stage7SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| generated_stage7::Stage7NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_generated_stage3_proof(
    artifacts: &Stage3ExecutionArtifacts<Fr>,
) -> generated_stage3::Stage3Proof<Fr> {
    generated_stage3::Stage3Proof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_generated_stage3_sumcheck_output)
            .collect(),
    }
}

fn to_generated_stage3_sumcheck_output(
    output: &jolt_kernels::stage3::Stage3SumcheckOutput<Fr>,
) -> generated_stage3::Stage3SumcheckOutput<Fr> {
    generated_stage3::Stage3SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| generated_stage3::Stage3NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
}

fn to_generated_stage4_proof(
    artifacts: &Stage4ExecutionArtifacts<Fr>,
) -> generated_stage4::Stage4Proof<Fr> {
    generated_stage4::Stage4Proof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(to_generated_stage4_sumcheck_output)
            .collect(),
    }
}

fn to_generated_stage4_sumcheck_output(
    output: &jolt_kernels::stage4::Stage4SumcheckOutput<Fr>,
) -> generated_stage4::Stage4SumcheckOutput<Fr> {
    generated_stage4::Stage4SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output
            .evals
            .iter()
            .map(|eval| generated_stage4::Stage4NamedEval {
                name: eval.name,
                oracle: eval.oracle,
                value: eval.value,
            })
            .collect(),
        proof: output.proof.clone(),
    }
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

fn assert_core_stage1_uniskip_proof_matches_bolt(
    proof: &CoreProof,
    output: &jolt_kernels::stage1::Stage1SumcheckOutput<Fr>,
) {
    let core_coefficients = match &proof.stage1_uni_skip_first_round_proof {
        UniSkipFirstRoundProofVariant::Standard(proof) => proof
            .uni_poly
            .coeffs
            .iter()
            .copied()
            .map(Fr::from)
            .collect::<Vec<_>>(),
        UniSkipFirstRoundProofVariant::Zk(_) => panic!("standard Stage 1 proof expected"),
    };
    assert_eq!(output.proof.round_polynomials.len(), 1);
    let bolt_coefficients = output.proof.round_polynomials[0].coefficients();
    if let Some(index) = bolt_coefficients
        .iter()
        .zip(core_coefficients.iter())
        .position(|(bolt, core)| bolt != core)
    {
        let ratio = if core_coefficients[index] != Fr::from_u64(0) {
            Some(bolt_coefficients[index] * core_coefficients[index].inverse().unwrap())
        } else {
            None
        };
        let next_ratio = bolt_coefficients
            .iter()
            .zip(core_coefficients.iter())
            .enumerate()
            .skip(index + 1)
            .find(|(_, (_, core))| **core != Fr::from_u64(0))
            .map(|(_, (bolt, core))| *bolt * core.inverse().unwrap());
        panic!(
            "Stage 1 uni-skip coefficient mismatch at {index}: bolt={:?} core={:?} ratio={:?} next_ratio={:?}",
            bolt_coefficients[index], core_coefficients[index], ratio, next_ratio
        );
    }
    assert_eq!(
        bolt_coefficients.len(),
        core_coefficients.len(),
        "Stage 1 uni-skip coefficient count mismatch"
    );
}

fn assert_stage1_uniskip_extended_evals_match_core(
    proof: &CoreProof,
    typed_data: &Stage1OuterRv64Data<'_>,
    core_row_data: &Stage1OuterRv64Data<'_>,
    generic_data: &Stage1OuterR1csData<'_, Fr>,
    artifacts: &Stage1ExecutionArtifacts<Fr>,
) {
    let tau = artifacts
        .challenge_vectors
        .iter()
        .find(|vector| vector.symbol == "stage1.tau")
        .expect("Bolt stage1 tau")
        .values
        .as_slice();
    let typed_evals = typed_data
        .uniskip_extended_evals(tau)
        .expect("typed Stage 1 extended evals");
    let generic_evals = generic_data
        .uniskip_extended_evals(tau)
        .expect("generic Stage 1 extended evals");
    assert_stage1_extended_eval_vecs_match(
        "typed RV64 vs generic R1CS",
        &typed_evals,
        &generic_evals,
    );

    let core_row_evals = core_row_data
        .uniskip_extended_evals(tau)
        .expect("core-row Stage 1 extended evals");
    let core_evals = core_stage1_uniskip_extended_evals(proof, tau[tau.len() - 1]);
    assert_stage1_extended_eval_vecs_match(
        "core-row RV64 vs jolt-core",
        &core_row_evals,
        &core_evals,
    );
    assert_stage1_extended_eval_vecs_match(
        "Bolt typed RV64 vs core-row RV64",
        &typed_evals,
        &core_row_evals,
    );
}

fn core_stage1_uniskip_extended_evals(proof: &CoreProof, tau_high: Fr) -> Vec<Fr> {
    let coefficients = match &proof.stage1_uni_skip_first_round_proof {
        UniSkipFirstRoundProofVariant::Standard(proof) => proof
            .uni_poly
            .coeffs
            .iter()
            .copied()
            .map(Fr::from)
            .collect::<Vec<_>>(),
        UniSkipFirstRoundProofVariant::Zk(_) => panic!("standard Stage 1 proof expected"),
    };
    let s1 = UnivariatePoly::new(coefficients);
    stage1_uniskip_targets()
        .into_iter()
        .map(|target| {
            let y = Fr::from_i64(target);
            let kernel = lagrange_kernel_eval(-4, 10, tau_high, y);
            s1.evaluate(y) / kernel
        })
        .collect()
}

fn assert_stage1_extended_eval_vecs_match(label: &str, actual: &[Fr], expected: &[Fr]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} extended eval count mismatch"
    );
    let targets = stage1_uniskip_targets();
    if let Some(index) = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual, expected)| actual != expected)
    {
        panic!(
            "{label} Stage 1 extended eval mismatch at target {} (index {index}): actual={:?} expected={:?}",
            targets[index], actual[index], expected[index]
        );
    }
}

fn stage1_uniskip_targets() -> [i64; 9] {
    [-5, 6, -6, 7, -7, 8, -8, 9, -9]
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

fn assert_core_stage3_sumcheck_proof_matches_bolt(
    proof: &CoreProof,
    output: &jolt_kernels::stage3::Stage3SumcheckOutput<Fr>,
) {
    let core_polys = match &proof.stage3_sumcheck_proof {
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(proof) => {
            &proof.compressed_polys
        }
        jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Zk(_) => {
            panic!("standard Stage 3 proof expected")
        }
    };
    assert_eq!(
        core_polys.len(),
        output.proof.round_polynomials.len(),
        "Stage 3 round count mismatch"
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
            "Stage 3 compressed coefficient mismatch at round {round}"
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

fn assert_core_preamble_states_match_bolt(
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

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert!(
        bolt_states.len() >= core_states.len(),
        "Bolt transcript shorter than core preamble"
    );
    for (index, (expected, actual)) in core_states.iter().zip(&bolt_states).enumerate() {
        assert_eq!(
            expected, actual,
            "pre-stage transcript state mismatch at op #{index}"
        );
    }
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

#[allow(dead_code)]
fn assert_core_states_match_bolt_stage3(
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_state_history_match(&core_states, &bolt_states);
}

fn assert_bolt_stage1_tamper_rejected(
    stage1_verifier_plan: &'static KernelStage1CpuProgramPlan,
    generated_stage1_verifier_plan: &'static generated_stage1::Stage1VerifierProgramPlan,
    proof: &Stage1Proof<Fr>,
    fixture: &CoreMuldivCommitmentFixture,
    commitment_verifier_trace: &BoltCommitmentTrace,
    transcript_steps: &[TranscriptStep],
) {
    let assert_stage1_tamper_rejected = |tampered: Stage1Proof<Fr>, message: &str| {
        let mut transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut transcript, fixture);
        append_bolt_commitments_to_transcript(
            &mut transcript,
            &commitment_verifier_trace.records,
            &commitment_verifier_trace.commitments,
            transcript_steps,
        );

        let mut verifier = Stage1VerifierKernelExecutor::new(&tampered);
        let result = execute_stage1_program(
            stage1_verifier_plan,
            Stage1ExecutionMode::Verifier,
            &mut verifier,
            &mut transcript,
        );
        assert!(result.is_err(), "{message}");

        let mut generated_transcript =
            CheckpointTranscript::<jolt_transcript::Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
        append_bolt_preamble(&mut generated_transcript, fixture);
        append_bolt_commitments_to_transcript(
            &mut generated_transcript,
            &commitment_verifier_trace.records,
            &commitment_verifier_trace.commitments,
            transcript_steps,
        );
        let generated_tampered = to_generated_stage1_proof(&tampered);
        let generated_result = generated_stage1::verify_stage1_outer_with_program(
            generated_stage1_verifier_plan,
            &generated_tampered,
            &mut generated_transcript,
        );
        assert!(generated_result.is_err(), "generated {message}");
    };

    let mut tampered_remaining_poly = proof.clone();
    let round_poly = &mut tampered_remaining_poly.sumchecks[1].proof.round_polynomials[0];
    let mut coefficients = round_poly.coefficients().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round_poly = UnivariatePoly::new(coefficients);
    assert_stage1_tamper_rejected(
        tampered_remaining_poly,
        "Bolt Stage 1 verifier accepted a tampered remaining sumcheck coefficient",
    );

    let mut tampered_uniskip_poly = proof.clone();
    let round_poly = &mut tampered_uniskip_poly.sumchecks[0].proof.round_polynomials[0];
    let mut coefficients = round_poly.coefficients().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round_poly = UnivariatePoly::new(coefficients);
    assert_stage1_tamper_rejected(
        tampered_uniskip_poly,
        "Bolt Stage 1 verifier accepted a tampered uniskip sumcheck coefficient",
    );

    let mut tampered_uniskip_point = proof.clone();
    tampered_uniskip_point.sumchecks[0].point[0] += Fr::from_u64(1);
    assert_stage1_tamper_rejected(
        tampered_uniskip_point,
        "Bolt Stage 1 verifier accepted a tampered uniskip point",
    );

    let mut tampered_uniskip_eval = proof.clone();
    tampered_uniskip_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    assert_stage1_tamper_rejected(
        tampered_uniskip_eval,
        "Bolt Stage 1 verifier accepted a tampered uniskip eval",
    );

    let mut tampered_remaining_point = proof.clone();
    tampered_remaining_point.sumchecks[1].point[0] += Fr::from_u64(1);
    assert_stage1_tamper_rejected(
        tampered_remaining_point,
        "Bolt Stage 1 verifier accepted a tampered remaining sumcheck point",
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

fn assert_state_history_prefix_match(expected_prefix: &[[u8; 32]], actual: &[[u8; 32]]) {
    assert!(
        actual.len() >= expected_prefix.len(),
        "transcript state count mismatch: expected at least {}, got {}",
        expected_prefix.len(),
        actual.len()
    );
    for index in 0..expected_prefix.len() {
        assert_eq!(
            expected_prefix[index], actual[index],
            "transcript state mismatch at op #{index}"
        );
    }
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
