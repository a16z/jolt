//! Bolt-side oracle driver for full real-trace equivalence gates.
//!
//! This module orchestrates the generated/kernel Bolt prover and verifier path
//! and compares its public artifacts against the core oracle. Stage semantics
//! should continue to move into Bolt, kernels, generated crates, or
//! `jolt-witness`, not into this module.

#![expect(
    clippy::expect_used,
    reason = "equivalence oracle should fail fast on invalid generated artifacts"
)]

use std::time::Instant;

use jolt_dory::{DoryProof, DoryScheme};
use jolt_openings::CommitmentScheme as _;
use jolt_profiling::{check_core_vs_bolt_gate, time_it, PeakRssSampler};

use crate::adapters::{
    canonical_generated_stage4_execution_artifacts, canonical_generated_stage5_execution_artifacts,
    canonical_generated_stage5_proof, canonical_generated_stage6_execution_artifacts,
    canonical_generated_stage6_proof, canonical_generated_stage7_execution_artifacts,
    canonical_generated_stage7_proof, canonical_stage4_artifacts,
};
use crate::artifacts::{ArtifactSource, EquivalenceRun, TranscriptTrace, VerifierResult};
use crate::bolt_programs::{
    bolt_commitment_programs_with_params, bolt_stage1_programs_with_params,
    bolt_stage2_programs_with_params, bolt_stage3_programs_with_params,
    bolt_stage4_programs_with_params, bolt_stage5_programs_with_params,
    bolt_stage6_programs_with_params, bolt_stage7_programs_with_params,
    bolt_stage8_programs_with_params,
};
use crate::checkpoint::{assert_state_history_match, assert_state_history_prefix_match};
use crate::checks::{
    assert_canonical_stage_artifacts_match, assert_dory_proofs_match,
    assert_equivalence_run_artifacts_match, assert_stage5_artifacts_match,
    assert_stage6_artifacts_match, assert_stage7_artifacts_match,
};
use crate::commitment_oracle::{
    generated_commitment_trace, generated_verifier_commitment_trace,
    run_generated_bolt_commitment_pair_with_cycles, transcript_with_bolt_commitment_trace,
    transcript_with_bolt_preamble, GeneratedCommitmentInputStorage,
};
use crate::core_oracle::{
    assert_core_accepts_bolt_evaluation_proof, assert_core_accepts_bolt_stage3,
    assert_core_accepts_bolt_stage4, assert_core_accepts_bolt_stage5,
    assert_core_accepts_bolt_stage6, assert_core_accepts_bolt_stage7,
    assert_core_accepts_full_bolt_proof, CoreMuldivCommitmentFixture,
};
use crate::perf::{
    generated_bolt_perf_metrics, print_core_vs_bolt_perf_summary, CORE_VS_BOLT_PERF_THRESHOLDS,
};
use crate::plan_adapters::{
    leak_generated_commitment_prover_program, leak_generated_commitment_verifier_program,
    leak_generated_stage1_verifier_program, leak_generated_stage2_verifier_program,
    leak_generated_stage3_verifier_program, leak_generated_stage4_verifier_program,
    leak_generated_stage5_verifier_program, leak_generated_stage6_verifier_program,
    leak_generated_stage7_verifier_program, leak_generated_stage8_prover_program,
    leak_generated_stage8_verifier_program, leak_stage1_program, leak_stage2_program,
    leak_stage3_program, leak_stage4_program, leak_stage5_program, leak_stage6_program,
    leak_stage7_program,
};
use crate::tamper::{
    assert_bolt_stage3_4_5_tamper_rejected, assert_bolt_stage6_tamper_rejected,
    assert_bolt_stage7_tamper_rejected, assert_monolithic_jolt_tamper_rejected,
    MonolithicJoltTamperInput, Stage345TamperInput, Stage6TamperInput, Stage7TamperInput,
};

pub fn assert_bolt_full_real_trace_self_parity(
    fixture: CoreMuldivCommitmentFixture,
    enforce_perf_gate: bool,
) {
    let bolt_setup_start = Instant::now();
    let _bolt_setup_span = tracing::info_span!("bolt.setup").entered();
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
    let generated_commitment_prover_plan =
        leak_generated_commitment_prover_program(&commitment_prover_program);
    let generated_commitment_verifier_plan =
        leak_generated_commitment_verifier_program(&commitment_verifier_program);

    let (commitment_prover_trace, commitment_verifier_trace) =
        run_generated_bolt_commitment_pair_with_cycles(
            generated_commitment_prover_plan,
            generated_commitment_verifier_plan,
            &fixture.pcs_setup,
            &fixture.cycle_inputs,
        );

    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let stage2_prover_plan = leak_stage2_program(&stage2_prover_program);
    let stage3_prover_plan = leak_stage3_program(&stage3_prover_program);
    let stage4_prover_plan = leak_stage4_program(&stage4_prover_program);
    let stage5_prover_plan = leak_stage5_program(&stage5_prover_program);
    let stage6_prover_plan = leak_stage6_program(&stage6_prover_program);
    let stage7_prover_plan = leak_stage7_program(&stage7_prover_program);
    let stage8_prover_plan = leak_generated_stage8_prover_program(&stage8_prover_program);
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
        stage6: generated_stage6_verifier_plan,
        stage7: generated_stage7_verifier_plan,
        stage8: generated_stage8_verifier_plan,
    };
    let r1cs_key = fixture.r1cs_key();
    let data = fixture.stage1_outer_rv64_data(&r1cs_key);
    let bolt_setup_ms = bolt_setup_start.elapsed().as_secs_f64() * 1_000.0;
    drop(_bolt_setup_span);

    let mut prover_transcript =
        transcript_with_bolt_commitment_trace(&fixture, &commitment_prover_trace);
    let stage1_artifacts = jolt_prover::prove_stage1_outer_with_witness_inputs(
        stage1_prover_plan,
        r1cs_key.num_cycle_vars(),
        &data,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");

    let stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(stage2_prover_plan, &stage1_artifacts)
            .expect("generated prover derives Stage 2 opening inputs from artifacts");
    let ram_data = fixture.stage2_ram_data();
    let stage2_artifacts = jolt_prover::prove_stage2_with_witness_inputs(
        stage2_prover_plan,
        &stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 2 prover succeeds");

    let stage3_openings = jolt_prover::stage3_opening_inputs_from_artifacts(
        stage3_prover_plan,
        &stage1_artifacts,
        &stage2_artifacts,
    )
    .expect("generated prover derives Stage 3 opening inputs from artifacts");
    let stage3_artifacts = jolt_prover::prove_stage3_with_witness_inputs(
        stage3_prover_plan,
        &stage3_openings,
        &fixture.stage3_cycles,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 3 prover succeeds");

    let stage4_openings = jolt_prover::stage4_opening_inputs_from_artifacts(
        stage4_prover_plan,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    )
    .expect("generated prover derives Stage 4 opening inputs from artifacts");
    let stage4_artifacts = jolt_prover::prove_stage4_with_trace_witness_inputs(
        stage4_prover_plan,
        &stage4_openings,
        1 << fixture.params.register_log_k,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 4 prover succeeds");

    let mut verifier_transcript =
        transcript_with_bolt_commitment_trace(&fixture, &commitment_verifier_trace);
    let stage1_proof = stage1_artifacts.clone().into();
    let verified_stage1 = jolt_prover::replay_stage1_outer_proof_with_program(
        stage1_prover_plan,
        &stage1_proof,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts");
    let stage2_verifier_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(stage2_prover_plan, &verified_stage1)
            .expect("generated prover derives Stage 2 verifier opening inputs from artifacts");
    let stage2_proof = stage2_artifacts.clone().into();
    let verified_stage2 = jolt_prover::replay_stage2_proof_with_program(
        stage2_prover_plan,
        &stage2_proof,
        &stage2_verifier_openings,
        Some(&ram_data),
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts");
    let generated_stage3_prefix_transcript = verifier_transcript.clone();
    let stage3_verifier_openings = jolt_prover::stage3_opening_inputs_from_artifacts(
        stage3_prover_plan,
        &verified_stage1,
        &verified_stage2,
    )
    .expect("generated prover derives Stage 3 verifier opening inputs from artifacts");
    let stage3_proof = stage3_artifacts.clone().into();
    let verified_stage3 = jolt_prover::replay_stage3_proof_with_program(
        stage3_prover_plan,
        &stage3_proof,
        &stage3_verifier_openings,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 3 verifier accepts");
    let stage4_verifier_openings = jolt_prover::stage4_opening_inputs_from_artifacts(
        stage4_prover_plan,
        &fixture.initial_ram_state,
        &verified_stage2,
        &verified_stage3,
    )
    .expect("generated prover derives Stage 4 verifier opening inputs from artifacts");
    let stage4_proof = stage4_artifacts.clone().into();
    let verified_stage4 = jolt_prover::replay_stage4_proof_with_program(
        stage4_prover_plan,
        &stage4_proof,
        &stage4_verifier_openings,
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
    assert_state_history_match(prover_transcript.log(), verifier_transcript.log());
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

    let mut generated_verifier_transcript = generated_stage3_prefix_transcript;
    let generated_stage3_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&stage3_verifier_openings);
    let generated_stage3_proof = jolt_prover::stage3_proof(&stage3_artifacts);
    let generated_stage3_start_transcript = generated_verifier_transcript.clone();
    let generated_verified_stage3 = jolt_verifier::verify_stage3_with_program(
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
    let generated_stage4_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&stage4_verifier_openings);
    let generated_stage4_proof = jolt_prover::stage4_proof(&stage4_artifacts);
    let generated_stage4_start_transcript = generated_verifier_transcript.clone();
    let generated_verified_stage4 = jolt_verifier::verify_stage4_with_program(
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
        verifier_transcript.log(),
        generated_verifier_transcript.log(),
    );
    let kernel_stage5_openings = jolt_prover::stage5_opening_inputs_from_artifacts(
        stage5_prover_plan,
        &stage2_artifacts,
        &stage4_artifacts,
    )
    .expect("generated prover derives Stage 5 opening inputs from artifacts");
    let generated_stage5_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&kernel_stage5_openings);
    let mut stage5_prover_transcript = prover_transcript.clone();
    let stage5_artifacts = jolt_prover::prove_stage5_with_trace_witness_inputs(
        stage5_prover_plan,
        &kernel_stage5_openings,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        1 << fixture.params.register_log_k,
        &fixture.stage5_lookup_indices,
        &fixture.stage5_lookup_table_indices,
        &fixture.stage5_is_interleaved_operands,
        fixture.params.lookups_ra_virtual_log_k_chunk,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut stage5_prover_transcript,
    )
    .expect("Bolt Stage 5 prover succeeds");
    assert_eq!(stage5_artifacts.sumchecks.len(), 1);
    let stage5_proof = jolt_prover::stage5_kernel_proof(&stage5_artifacts);
    let mut kernel_stage5_transcript = verifier_transcript.clone();
    let kernel_verified_stage5 = jolt_prover::replay_stage5_proof_with_program(
        stage5_prover_plan,
        &stage5_proof,
        &kernel_stage5_openings,
        &mut kernel_stage5_transcript,
    )
    .expect("kernel Stage 5 replay accepts Bolt real muldiv proof");
    assert_eq!(kernel_verified_stage5.sumchecks.len(), 1);
    assert_state_history_match(
        stage5_prover_transcript.log(),
        kernel_stage5_transcript.log(),
    );
    let generated_stage5_proof = jolt_prover::stage5_proof(&stage5_artifacts);
    let generated_stage5_start_transcript = generated_verifier_transcript.clone();
    let generated_verified_stage5 = jolt_verifier::verify_stage5_with_program(
        generated_stage5_verifier_plan,
        &generated_stage5_proof,
        &generated_stage5_openings,
        &mut generated_verifier_transcript,
    )
    .expect("generated Stage 5 verifier accepts Bolt real muldiv proof");
    assert_eq!(generated_verified_stage5.sumchecks.len(), 1);
    let generated_verified_stage5_proof = jolt_verifier::JoltStageProof {
        sumchecks: generated_verified_stage5.sumchecks.clone(),
    };
    assert_stage5_artifacts_match(&generated_stage5_proof, &generated_verified_stage5_proof);
    assert_state_history_match(
        stage5_prover_transcript.log(),
        generated_verifier_transcript.log(),
    );
    assert_core_accepts_bolt_stage5(
        &fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
    );
    let kernel_stage6_openings = jolt_prover::stage6_opening_inputs_from_artifacts(
        stage6_prover_plan,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &stage5_artifacts,
    )
    .expect("generated prover derives Stage 6 opening inputs from artifacts");
    let generated_stage6_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&kernel_stage6_openings);
    let generated_stage6_data = jolt_prover::stage6_verifier_data_from_witness_entries(
        &fixture.stage6_bytecode_entries,
        fixture.stage6_entry_bytecode_index,
        fixture.stage6_num_lookup_tables,
    );
    let kernel_stage6_bytecode_data =
        jolt_prover::stage6_bytecode_read_raf_data_from_witness_entries(
            &fixture.stage6_bytecode_entries,
            fixture.stage6_entry_bytecode_index,
            fixture.stage6_num_lookup_tables,
        );
    let mut stage6_prover_transcript = generated_verifier_transcript.clone();
    let stage6_artifacts = jolt_prover::prove_stage6_with_trace_witness_inputs(
        stage6_prover_plan,
        &kernel_stage6_openings,
        kernel_stage6_bytecode_data.as_input(),
        fixture.stage6_witness_params(),
        &fixture.cycle_inputs,
        fixture.params.instruction_ra_virtual_d,
        &mut stage6_prover_transcript,
    )
    .expect("Bolt Stage 6 prover succeeds");
    assert_eq!(stage6_artifacts.sumchecks.len(), 1);
    let generated_stage6_proof = jolt_prover::stage6_proof(&stage6_artifacts);
    let generated_stage6_artifacts = jolt_prover::stage6_execution_artifacts(&stage6_artifacts);
    assert_stage6_artifacts_match(&generated_stage6_proof, &generated_stage6_artifacts);
    let mut kernel_stage6_transcript = generated_verifier_transcript.clone();
    let kernel_stage6_proof = jolt_prover::stage6_kernel_proof(&generated_stage6_proof);
    let kernel_verified_stage6 = jolt_prover::replay_stage6_proof_with_program(
        stage6_prover_plan,
        &kernel_stage6_proof,
        &kernel_stage6_openings,
        Some(kernel_stage6_bytecode_data.as_input()),
        &mut kernel_stage6_transcript,
    )
    .expect("kernel Stage 6 replay accepts Bolt real muldiv proof");
    assert_eq!(kernel_verified_stage6.sumchecks.len(), 1);
    assert_stage6_artifacts_match(
        &generated_stage6_proof,
        &jolt_prover::stage6_execution_artifacts(&kernel_verified_stage6),
    );
    assert_state_history_match(
        stage6_prover_transcript.log(),
        kernel_stage6_transcript.log(),
    );
    let mut generated_stage6_transcript = generated_verifier_transcript.clone();
    let generated_verified_stage6 = jolt_verifier::verify_stage6_with_program(
        generated_stage6_verifier_plan,
        &generated_stage6_proof,
        &generated_stage6_openings,
        Some(&generated_stage6_data),
        &mut generated_stage6_transcript,
    )
    .expect("generated Stage 6 verifier accepts Bolt real muldiv proof");
    assert_stage6_artifacts_match(&generated_stage6_proof, &generated_verified_stage6);
    assert_state_history_match(
        stage6_prover_transcript.log(),
        generated_stage6_transcript.log(),
    );
    assert_core_accepts_bolt_stage6(
        &fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
        &generated_stage6_proof,
    );

    let kernel_stage7_openings =
        jolt_prover::stage7_opening_inputs_from_stage6_artifacts_with_program(
            stage7_prover_plan,
            &stage6_artifacts,
        )
        .expect("generated prover derives Stage 7 opening inputs from Stage 6 artifacts");
    let generated_stage7_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&kernel_stage7_openings);
    let mut stage7_prover_transcript = stage6_prover_transcript.clone();
    let stage7_artifacts = jolt_prover::prove_stage7_with_trace_witness_inputs(
        stage7_prover_plan,
        &kernel_stage7_openings,
        fixture.stage6_witness_params(),
        &fixture.cycle_inputs,
        &kernel_stage6_openings,
        &mut stage7_prover_transcript,
    )
    .expect("Bolt Stage 7 prover succeeds");
    assert_eq!(stage7_artifacts.sumchecks.len(), 1);
    let generated_stage7_proof = jolt_prover::stage7_proof(&stage7_artifacts);
    let generated_stage7_artifacts = jolt_prover::stage7_execution_artifacts(&stage7_artifacts);
    assert_stage7_artifacts_match(&generated_stage7_proof, &generated_stage7_artifacts);
    let mut kernel_stage7_transcript = generated_stage6_transcript.clone();
    let kernel_stage7_proof = jolt_prover::stage7_kernel_proof(&generated_stage7_proof);
    let kernel_verified_stage7 = jolt_prover::replay_stage7_proof_with_program(
        stage7_prover_plan,
        &kernel_stage7_proof,
        &kernel_stage7_openings,
        &mut kernel_stage7_transcript,
    )
    .expect("kernel Stage 7 replay accepts Bolt real muldiv proof");
    assert_eq!(kernel_verified_stage7.sumchecks.len(), 1);
    assert_stage7_artifacts_match(
        &generated_stage7_proof,
        &jolt_prover::stage7_execution_artifacts(&kernel_verified_stage7),
    );
    assert_state_history_match(
        stage7_prover_transcript.log(),
        kernel_stage7_transcript.log(),
    );
    let mut generated_stage7_transcript = generated_stage6_transcript.clone();
    let generated_verified_stage7 = jolt_verifier::verify_stage7_with_program(
        generated_stage7_verifier_plan,
        &generated_stage7_proof,
        &generated_stage7_openings,
        &mut generated_stage7_transcript,
    )
    .expect("generated Stage 7 verifier accepts Bolt real muldiv proof");
    assert_stage7_artifacts_match(&generated_stage7_proof, &generated_verified_stage7);
    assert_state_history_match(
        stage7_prover_transcript.log(),
        generated_stage7_transcript.log(),
    );
    assert_core_accepts_bolt_stage7(
        &fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
        &generated_stage6_proof,
        &generated_stage7_proof,
    );

    let generated_jolt_stage2_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&stage2_verifier_openings);
    let generated_jolt_stage3_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&stage3_verifier_openings);
    let generated_jolt_stage4_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&stage4_verifier_openings);
    let generated_ram_data_storage = jolt_prover::stage2_verifier_ram_data(&ram_data);
    let generated_ram_data = generated_ram_data_storage.as_input();
    let generated_jolt_inputs = jolt_verifier::JoltVerifierInputs {
        stage2_openings: &generated_jolt_stage2_openings,
        stage2_ram: Some(&generated_ram_data),
        stage3_openings: &generated_jolt_stage3_openings,
        stage4_openings: &generated_jolt_stage4_openings,
        stage5_openings: &generated_stage5_openings,
        stage6_openings: &generated_stage6_openings,
        stage6_data: Some(&generated_stage6_data),
        stage7_openings: &generated_stage7_openings,
        evaluation_setup: None,
    };
    let generated_jolt_proof = jolt_prover::jolt_proof_through_stage5(
        &commitment_verifier_trace.commitments,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
    );
    let mut generated_jolt_transcript = transcript_with_bolt_preamble(&fixture);
    let generated_jolt_artifacts = jolt_verifier::verify_jolt_through_stage5_with_programs(
        &generated_jolt_proof,
        generated_jolt_inputs.through_stage5(),
        generated_programs,
        &mut generated_jolt_transcript,
    )
    .expect("generated monolithic verifier accepts real muldiv proof");
    assert_eq!(
        stage3_artifacts.sumchecks.len(),
        generated_jolt_artifacts.stage3.sumchecks.len()
    );
    assert_canonical_stage_artifacts_match(
        "generated monolithic through-stage5 Stage 4",
        canonical_stage4_artifacts(&stage4_artifacts),
        canonical_generated_stage4_execution_artifacts(&generated_jolt_artifacts.stage4),
    );
    assert_canonical_stage_artifacts_match(
        "generated monolithic through-stage5 Stage 5",
        canonical_generated_stage5_proof(&generated_stage5_proof),
        canonical_generated_stage5_execution_artifacts(&generated_jolt_artifacts.stage5),
    );
    assert_state_history_match(
        generated_verifier_transcript.log(),
        generated_jolt_transcript.log(),
    );
    let generated_jolt_proof_with_stage6 = jolt_prover::jolt_proof_through_stage6(
        &commitment_verifier_trace.commitments,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
        &generated_stage6_proof,
    );
    let mut generated_jolt_stage6_transcript = transcript_with_bolt_preamble(&fixture);
    let generated_jolt_stage6_artifacts = jolt_verifier::verify_jolt_through_stage6_with_programs(
        &generated_jolt_proof_with_stage6,
        generated_jolt_inputs.through_stage6(),
        generated_programs,
        &mut generated_jolt_stage6_transcript,
    )
    .expect("generated monolithic verifier accepts Bolt Stage 6 proof");
    assert_stage6_artifacts_match(
        &generated_stage6_proof,
        &generated_jolt_stage6_artifacts.stage6,
    );
    assert_state_history_match(
        generated_stage6_transcript.log(),
        generated_jolt_stage6_transcript.log(),
    );
    let generated_jolt_proof_with_stage7 = jolt_prover::jolt_proof_through_stage7(
        &commitment_verifier_trace.commitments,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &generated_stage5_proof,
        &generated_stage6_proof,
        &generated_stage7_proof,
    );
    let mut generated_jolt_stage7_transcript = transcript_with_bolt_preamble(&fixture);
    let generated_jolt_stage7_artifacts = jolt_verifier::verify_jolt_through_stage7_with_programs(
        &generated_jolt_proof_with_stage7,
        generated_jolt_inputs.through_stage7(),
        generated_programs,
        &mut generated_jolt_stage7_transcript,
    )
    .expect("generated monolithic verifier accepts Bolt Stage 7 proof");
    assert_stage7_artifacts_match(
        &generated_stage7_proof,
        &generated_jolt_stage7_artifacts.stage7,
    );
    assert_state_history_match(
        generated_stage7_transcript.log(),
        generated_jolt_stage7_transcript.log(),
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
    let monolithic_commitment_storage =
        GeneratedCommitmentInputStorage::from_cycles(&fixture.cycle_inputs);
    let mut monolithic_commitment_inputs = monolithic_commitment_storage.sparse_inputs();
    let mut monolithic_prover_transcript = transcript_with_bolt_preamble(&fixture);
    let bolt_rss_sampler = PeakRssSampler::start().expect("start Bolt RSS sampler");
    let (bolt_prove_ms, monolithic_prove_result) = time_it(|| {
        jolt_prover::prove_jolt_with_witness_inputs(
            jolt_prover::JoltProverWitnessInputs {
                commitment_inputs: &mut monolithic_commitment_inputs,
                prover_setup: &fixture.pcs_setup,
                stage1_trace_num_vars: r1cs_key.num_cycle_vars(),
                stage1_outer_evaluator: &data,
                stage2_openings: &stage2_openings,
                product_virtual_cycles: &fixture.product_virtual_cycles,
                instruction_lookup_cycles: &fixture.instruction_lookup_cycles,
                ram: &ram_data,
                stage3_openings: &stage3_openings,
                stage3_cycles: &fixture.stage3_cycles,
                stage4_openings: &stage4_openings,
                register_count: 1 << fixture.params.register_log_k,
                trace_len: fixture.proof.trace_length,
                ram_k: fixture.proof.ram_K,
                register_accesses: &fixture.stage4_register_accesses,
                stage5_openings: &kernel_stage5_openings,
                lookup_indices: &fixture.stage5_lookup_indices,
                lookup_table_indices: &fixture.stage5_lookup_table_indices,
                is_interleaved_operands: &fixture.stage5_is_interleaved_operands,
                ra_virtual_log_k_chunk: fixture.params.lookups_ra_virtual_log_k_chunk,
                stage6_openings: &kernel_stage6_openings,
                stage6_bytecode_data: kernel_stage6_bytecode_data.as_input(),
                stage6_witness_params: fixture.stage6_witness_params(),
                cycle_inputs: &fixture.cycle_inputs,
                instruction_ra_virtual_d: fixture.params.instruction_ra_virtual_d,
                stage7_openings: &kernel_stage7_openings,
                evaluation_openings: Some(&kernel_stage7_openings),
            },
            monolithic_prover_programs,
            &mut monolithic_prover_transcript,
        )
    });
    let bolt_peak_rss_mb = bolt_rss_sampler.finish();
    let (monolithic_proof, monolithic_artifacts) =
        monolithic_prove_result.expect("generated monolithic prover produces real trace proof");
    let monolithic_evaluation = monolithic_proof
        .evaluation
        .as_ref()
        .expect("generated monolithic prover emits evaluation proof");
    assert_state_history_prefix_match(
        generated_stage7_transcript.log(),
        monolithic_prover_transcript.log(),
    );
    assert_dory_proofs_match(
        &DoryProof(fixture.proof.joint_opening_proof.clone()),
        &monolithic_evaluation.joint_opening_proof,
    );
    assert_core_accepts_full_bolt_proof(&fixture, &monolithic_proof, &monolithic_artifacts);
    assert_core_accepts_bolt_evaluation_proof(&fixture, monolithic_evaluation);

    let mut staged_bolt_run = EquivalenceRun::new(ArtifactSource::Bolt);
    staged_bolt_run.commitments = commitment_prover_trace.commitment_trace();
    staged_bolt_run.stages = vec![
        canonical_stage4_artifacts(&stage4_artifacts),
        canonical_generated_stage5_proof(&generated_stage5_proof),
        canonical_generated_stage6_proof(&generated_stage6_proof),
        canonical_generated_stage7_proof(&generated_stage7_proof),
    ];
    staged_bolt_run.verifier_result = VerifierResult::accepted();
    let monolithic_stage5_proof = jolt_prover::stage5_proof(&monolithic_artifacts.stage5);
    let monolithic_stage6_artifacts =
        jolt_prover::stage6_execution_artifacts(&monolithic_artifacts.stage6);
    let monolithic_stage7_artifacts =
        jolt_prover::stage7_execution_artifacts(&monolithic_artifacts.stage7);
    let mut monolithic_bolt_run = EquivalenceRun::new(ArtifactSource::Bolt);
    monolithic_bolt_run.commitments = generated_commitment_trace(&monolithic_artifacts.commitment);
    monolithic_bolt_run.transcript = TranscriptTrace {
        events: monolithic_prover_transcript.log().to_vec(),
    };
    monolithic_bolt_run.stages = vec![
        canonical_stage4_artifacts(&monolithic_artifacts.stage4),
        canonical_generated_stage5_proof(&monolithic_stage5_proof),
        canonical_generated_stage6_execution_artifacts(&monolithic_stage6_artifacts),
        canonical_generated_stage7_execution_artifacts(&monolithic_stage7_artifacts),
    ];
    monolithic_bolt_run.verifier_result = VerifierResult::accepted();
    assert_equivalence_run_artifacts_match(
        "staged-vs-monolithic generated prover",
        &staged_bolt_run,
        &monolithic_bolt_run,
    );

    let mut monolithic_verify_transcript = transcript_with_bolt_preamble(&fixture);
    let evaluation_setup = DoryScheme::verifier_setup(&fixture.pcs_setup);
    let (bolt_verify_ms, monolithic_verify_result) = time_it(|| {
        jolt_verifier::verify_jolt_with_programs(
            &monolithic_proof,
            generated_jolt_inputs.full(&evaluation_setup),
            generated_programs,
            &mut monolithic_verify_transcript,
        )
    });
    let monolithic_verified_artifacts = monolithic_verify_result
        .expect("generated monolithic verifier accepts generated monolithic prover proof");
    let mut monolithic_verifier_run = EquivalenceRun::new(ArtifactSource::Bolt);
    monolithic_verifier_run.commitments =
        generated_verifier_commitment_trace(&monolithic_verified_artifacts.commitment);
    monolithic_verifier_run.transcript = TranscriptTrace {
        events: monolithic_verify_transcript.log().to_vec(),
    };
    monolithic_verifier_run.stages = vec![
        canonical_generated_stage4_execution_artifacts(&monolithic_verified_artifacts.stage4),
        canonical_generated_stage5_execution_artifacts(&monolithic_verified_artifacts.stage5),
        canonical_generated_stage6_execution_artifacts(&monolithic_verified_artifacts.stage6),
        canonical_generated_stage7_execution_artifacts(&monolithic_verified_artifacts.stage7),
    ];
    monolithic_verifier_run.verifier_result = VerifierResult::accepted();
    assert_equivalence_run_artifacts_match(
        "generated monolithic prover-vs-verifier",
        &monolithic_bolt_run,
        &monolithic_verifier_run,
    );

    let bolt_metrics = generated_bolt_perf_metrics(
        bolt_setup_ms,
        bolt_prove_ms,
        bolt_verify_ms,
        &monolithic_proof,
        bolt_peak_rss_mb,
    );
    if enforce_perf_gate {
        let report = check_core_vs_bolt_gate(
            &fixture.core_metrics,
            &bolt_metrics,
            CORE_VS_BOLT_PERF_THRESHOLDS,
        )
        .expect("core-vs-Bolt perf oracle gate");
        print_core_vs_bolt_perf_summary(&fixture.core_metrics, &bolt_metrics, &report);
    }
    assert_state_history_match(
        monolithic_prover_transcript.log(),
        monolithic_verify_transcript.log(),
    );

    assert_monolithic_jolt_tamper_rejected(MonolithicJoltTamperInput {
        preamble: &fixture,
        proof: &monolithic_proof,
        inputs: generated_jolt_inputs.full(&evaluation_setup),
        programs: generated_programs,
    });

    assert_bolt_stage3_4_5_tamper_rejected(Stage345TamperInput {
        preamble: &fixture,
        commitment_verifier_trace: &commitment_verifier_trace,
        stage1_artifacts: &stage1_artifacts,
        stage2_artifacts: &stage2_artifacts,
        stage3_artifacts: &stage3_artifacts,
        stage4_artifacts: &stage4_artifacts,
        generated_stage3_verifier_plan,
        generated_stage3_openings: &generated_stage3_openings,
        generated_stage3_start_transcript: &generated_stage3_start_transcript,
        generated_stage4_verifier_plan,
        generated_stage5_verifier_plan,
        generated_stage4_start_transcript: &generated_stage4_start_transcript,
        generated_stage4_openings: &generated_stage4_openings,
        generated_stage5_openings: &generated_stage5_openings,
        generated_stage5_proof: &generated_stage5_proof,
        generated_stage5_start_transcript: &generated_stage5_start_transcript,
        generated_jolt_inputs,
        generated_programs,
    });

    assert_bolt_stage6_tamper_rejected(Stage6TamperInput {
        preamble: &fixture,
        commitment_verifier_trace: &commitment_verifier_trace,
        verifier_transcript: &generated_verifier_transcript,
        verifier_plan: generated_stage6_verifier_plan,
        proof: &generated_stage6_proof,
        openings: &generated_stage6_openings,
        data: &generated_stage6_data,
        stage1_artifacts: &stage1_artifacts,
        stage2_artifacts: &stage2_artifacts,
        stage3_artifacts: &stage3_artifacts,
        stage4_artifacts: &stage4_artifacts,
        stage5_proof: &generated_stage5_proof,
        jolt_inputs: generated_jolt_inputs.through_stage6(),
        programs: generated_programs,
    });

    assert_bolt_stage7_tamper_rejected(Stage7TamperInput {
        preamble: &fixture,
        commitment_verifier_trace: &commitment_verifier_trace,
        verifier_transcript: &generated_stage6_transcript,
        verifier_plan: generated_stage7_verifier_plan,
        proof: &generated_stage7_proof,
        openings: &generated_stage7_openings,
        stage1_artifacts: &stage1_artifacts,
        stage2_artifacts: &stage2_artifacts,
        stage3_artifacts: &stage3_artifacts,
        stage4_artifacts: &stage4_artifacts,
        stage5_proof: &generated_stage5_proof,
        stage6_proof: &generated_stage6_proof,
        jolt_inputs: generated_jolt_inputs.through_stage7(),
        programs: generated_programs,
    });
}
