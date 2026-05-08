//! Commitment-phase transcript bridge between Bolt IR and jolt-core.
//!
//! This keeps the first oracle intentionally narrow: Bolt owns the commitment
//! ordering through its CPU IR, while jolt-core owns the reference
//! `append_serializable` transcript semantics for the same Dory commitments.

#![expect(
    clippy::expect_used,
    reason = "integration gates should fail fast when positive prover/verifier setup fails"
)]

use jolt_equivalence::bolt_oracle::assert_bolt_full_real_trace_self_parity;
use jolt_equivalence::bolt_programs::{
    bolt_commitment_programs, bolt_commitment_programs_with_params,
    bolt_stage1_programs_with_params, bolt_stage2_programs_with_params,
};
use jolt_equivalence::checkpoint::{assert_state_history_match, assert_transcripts_match};
use jolt_equivalence::checks::{
    assert_core_stage2_uniskip_proof_matches_bolt, assert_stage1_uniskip_extended_evals_match_core,
};
use jolt_equivalence::commitment_oracle::{
    core_commitment_trace, run_generated_bolt_commitment_pair_with_cpu_programs,
    run_generated_synthetic_bolt_commitment_pair_with_cpu_programs,
    transcript_with_bolt_commitment_trace,
};
use jolt_equivalence::core_conversion::{core_commitment_log, core_commitments_transcript_log};
use jolt_equivalence::core_oracle::{
    assert_core_accepts_bolt_stage1, assert_core_accepts_bolt_stage2,
    core_muldiv_commitment_fixture,
};
use jolt_equivalence::plan_adapters::{
    leak_generated_stage1_verifier_program, leak_generated_stage2_verifier_program,
    leak_stage1_program, leak_stage2_program,
};
use jolt_equivalence::tamper::{
    assert_bolt_chain_verifier_accepts_stage2_product_uniskip, assert_bolt_stage1_tamper_rejected,
    assert_bolt_stage2_batched_tamper_rejected, BoltStage2ChainVerifierInput,
    Stage2BatchedTamperInput,
};
use jolt_equivalence::ArtifactSource;

#[test]
fn bolt_commitment_transcript_matches_jolt_core_append_serializable() {
    let (prover_program, verifier_program) = bolt_commitment_programs();
    let (prover_trace, verifier_trace) =
        run_generated_synthetic_bolt_commitment_pair_with_cpu_programs(
            &prover_program,
            &verifier_program,
        );
    let core_log = core_commitment_log(
        prover_trace
            .records
            .iter()
            .zip(&prover_trace.commitments)
            .map(|(record, commitment)| (record.artifact.as_str(), commitment.as_ref())),
        &prover_program.transcript_steps,
    );

    let prover_run = prover_trace.equivalence_run(ArtifactSource::Bolt);
    let verifier_run = verifier_trace.equivalence_run(ArtifactSource::Bolt);
    assert_eq!(prover_run.commitments, verifier_run.commitments);
    assert_transcripts_match(&prover_trace.log, &verifier_trace.log);
    assert_eq!(prover_run.transcript, verifier_run.transcript);
    assert_transcripts_match(&core_log, &prover_trace.log);
}

#[test]
fn bolt_commitment_real_muldiv_trace_matches_jolt_core() {
    let fixture = core_muldiv_commitment_fixture();
    let (prover_program, verifier_program) = bolt_commitment_programs_with_params(&fixture.params);

    let (prover_trace, verifier_trace) = run_generated_bolt_commitment_pair_with_cpu_programs(
        &prover_program,
        &verifier_program,
        &fixture.pcs_setup,
        &fixture.cycle_inputs,
    );

    assert_eq!(
        prover_trace.committed_prefix(fixture.commitments.len()),
        core_commitment_trace(&fixture.commitments, "jolt.main_witness_commitments"),
    );

    let core_log =
        core_commitments_transcript_log(&fixture.commitments, &prover_program.transcript_steps);
    let prover_run = prover_trace.equivalence_run(ArtifactSource::Bolt);
    let verifier_run = verifier_trace.equivalence_run(ArtifactSource::Bolt);
    assert_eq!(prover_run.commitments, verifier_run.commitments);
    assert_transcripts_match(&prover_trace.log, &verifier_trace.log);
    assert_eq!(prover_run.transcript, verifier_run.transcript);
    assert_transcripts_match(&core_log, &prover_trace.log);
}

#[test]
fn bolt_commitment_stage1_real_muldiv_parity_checks() {
    let fixture = core_muldiv_commitment_fixture();
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, stage1_verifier_program) =
        bolt_stage1_programs_with_params(&fixture.params);

    let (commitment_prover_trace, commitment_verifier_trace) =
        run_generated_bolt_commitment_pair_with_cpu_programs(
            &commitment_prover_program,
            &commitment_verifier_program,
            &fixture.pcs_setup,
            &fixture.cycle_inputs,
        );

    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let generated_stage1_verifier_plan =
        leak_generated_stage1_verifier_program(&stage1_verifier_program);
    let r1cs_key = fixture.r1cs_key();
    let data = fixture.stage1_outer_rv64_data(&r1cs_key);
    let generic_data = fixture.stage1_outer_r1cs_data(&r1cs_key);

    let mut prover_transcript =
        transcript_with_bolt_commitment_trace(&fixture, &commitment_prover_trace);
    let stage1_artifacts = jolt_prover::prove_stage1_outer_with_witness_inputs(
        stage1_prover_plan,
        r1cs_key.num_cycle_vars(),
        &data,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");
    assert_stage1_uniskip_extended_evals_match_core(
        &fixture.proof,
        &data,
        &generic_data,
        &stage1_artifacts,
    );

    let stage1_proof = stage1_artifacts.clone().into();
    let mut verifier_transcript =
        transcript_with_bolt_commitment_trace(&fixture, &commitment_verifier_trace);
    let stage1_start_transcript = verifier_transcript.clone();
    let verified_stage1 = jolt_prover::replay_stage1_outer_proof_with_program(
        stage1_prover_plan,
        &stage1_proof,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts prover proof");

    assert_eq!(
        stage1_artifacts.sumchecks.len(),
        verified_stage1.sumchecks.len()
    );
    assert_transcripts_match(prover_transcript.log(), verifier_transcript.log());

    assert_core_accepts_bolt_stage1(&fixture, &stage1_artifacts);
    assert_bolt_stage1_tamper_rejected(
        stage1_prover_plan,
        generated_stage1_verifier_plan,
        &stage1_proof,
        &stage1_start_transcript,
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

    let (commitment_prover_trace, commitment_verifier_trace) =
        run_generated_bolt_commitment_pair_with_cpu_programs(
            &commitment_prover_program,
            &commitment_verifier_program,
            &fixture.pcs_setup,
            &fixture.cycle_inputs,
        );

    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let r1cs_key = fixture.r1cs_key();
    let data = fixture.stage1_outer_rv64_data(&r1cs_key);

    let mut bolt_transcript =
        transcript_with_bolt_commitment_trace(&fixture, &commitment_prover_trace);
    let stage1_artifacts = jolt_prover::prove_stage1_outer_with_witness_inputs(
        stage1_prover_plan,
        r1cs_key.num_cycle_vars(),
        &data,
        &mut bolt_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");

    let stage2_prover_plan = leak_stage2_program(&stage2_prover_program);
    let stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(stage2_prover_plan, &stage1_artifacts)
            .expect("generated prover derives Stage 2 opening inputs from artifacts");
    let ram_data = fixture.stage2_ram_data();
    let generated_stage2_verifier_plan =
        leak_generated_stage2_verifier_program(&stage2_verifier_program);
    let stage2_artifacts = jolt_prover::prove_stage2_with_witness_inputs(
        stage2_prover_plan,
        &stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut bolt_transcript,
    )
    .expect("Bolt Stage 2 prover succeeds");

    assert_core_stage2_uniskip_proof_matches_bolt(&fixture.proof, &stage2_artifacts.sumchecks[0]);
    assert_bolt_chain_verifier_accepts_stage2_product_uniskip(BoltStage2ChainVerifierInput {
        fixture: &fixture,
        commitment_verifier_trace: &commitment_verifier_trace,
        stage1_prover_plan,
        stage2_prover_plan,
        generated_stage2_verifier_plan,
        stage1_artifacts: &stage1_artifacts,
        stage2_artifacts: &stage2_artifacts,
        ram_data: &ram_data,
        prover_transcript: &bolt_transcript,
    });

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

    let (commitment_prover_trace, commitment_verifier_trace) =
        run_generated_bolt_commitment_pair_with_cpu_programs(
            &commitment_prover_program,
            &commitment_verifier_program,
            &fixture.pcs_setup,
            &fixture.cycle_inputs,
        );

    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let r1cs_key = fixture.r1cs_key();
    let data = fixture.stage1_outer_rv64_data(&r1cs_key);

    let mut prover_transcript =
        transcript_with_bolt_commitment_trace(&fixture, &commitment_prover_trace);
    let stage1_artifacts = jolt_prover::prove_stage1_outer_with_witness_inputs(
        stage1_prover_plan,
        r1cs_key.num_cycle_vars(),
        &data,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");

    let stage2_prover_plan = leak_stage2_program(&stage2_prover_program);
    let stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(stage2_prover_plan, &stage1_artifacts)
            .expect("generated prover derives Stage 2 opening inputs from artifacts");
    let ram_data = fixture.stage2_ram_data();
    let generated_stage2_verifier_plan =
        leak_generated_stage2_verifier_program(&stage2_verifier_program);
    let stage2_artifacts = jolt_prover::prove_stage2_with_witness_inputs(
        stage2_prover_plan,
        &stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 2 prover succeeds");

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
    let stage2_start_transcript = verifier_transcript.clone();
    let verified_stage2 = jolt_prover::replay_stage2_proof_with_program(
        stage2_prover_plan,
        &stage2_proof,
        &stage2_verifier_openings,
        Some(&ram_data),
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts");

    assert_eq!(
        stage2_artifacts.sumchecks.len(),
        verified_stage2.sumchecks.len()
    );
    assert_state_history_match(prover_transcript.log(), verifier_transcript.log());
    assert_core_accepts_bolt_stage2(&fixture, &stage1_artifacts, &stage2_artifacts);

    assert_bolt_stage2_batched_tamper_rejected(Stage2BatchedTamperInput {
        stage2_prover_plan,
        generated_stage2_verifier_plan,
        stage2_start_transcript: &stage2_start_transcript,
        stage2_openings: &stage2_verifier_openings,
        stage2_artifacts: &stage2_artifacts,
        ram_data: &ram_data,
    });
}

#[test]
#[ignore = "Stage 5+ instruction-read-RAF parity needs the follow-up generated kernel rewrite"]
fn bolt_stage3_batched_real_muldiv_self_parity() {
    assert_bolt_full_real_trace_self_parity(core_muldiv_commitment_fixture(), false);
}
