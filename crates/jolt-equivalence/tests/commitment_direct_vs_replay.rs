//! DIAGNOSTIC: isolate whether the commitment-style structural difference
//! between `bolt_oracle` (record-then-replay) and `jolt_host::prove_program`
//! (direct call against preamble-loaded transcript) is what causes
//! `prove_program`'s Stage 6 input-claim mismatch at log_t=9.
//!
//! Both variants:
//!   * Use the SAME fixture data (`core_muldiv_commitment_fixture_at_log_t(9)`)
//!   * Use the SAME goldens-committed program plans (`default_prover_programs`)
//!   * Use `ProveProgramPreambleSource` (zero `preprocessing_digest`) so the
//!     preamble matches what `prove_program` writes today
//!
//! Variant A: REPLAY style.
//!   1. Run the commitment phase against a FRESH transcript via
//!      `run_generated_bolt_commitment_pair_with_cycles` to record a
//!      `BoltCommitmentTrace`.
//!   2. Build the prover transcript by calling
//!      `transcript_with_bolt_commitment_trace` — preamble + replayed
//!      commitment-trace bytes.
//!   3. Drive Stages 1-6 against that transcript.
//!
//! Variant B: DIRECT style (mirrors `jolt_host::prove_program`).
//!   1. Build a preamble-loaded transcript via `BoltTranscript::new` +
//!      `append_bolt_preamble`.
//!   2. Call `prove_commitment_phase_with_program` DIRECTLY against that
//!      preamble-loaded transcript (no replay buffer).
//!   3. Drive Stages 1-6 against the SAME transcript.
//!
//! Both variants share the same fixture/programs, so the ONLY structural
//! difference is the commitment-style (replay vs direct). If Variant B
//! fails at Stage 6 while Variant A passes, the replay-vs-direct path
//! IS the bug behind `prove_program`. If both pass, the bug must lie in
//! the LIVE-trace data construction inside `prove_program` (something
//! drifting from the fixture — cycle_inputs ordering, padding, bytecode
//! preprocessing, ram_K rounding, etc.).

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::too_many_lines,
    unfulfilled_lint_expectations,
    reason = "diagnostic test fails fast on internal mismatches"
)]

use common::jolt_device::JoltDevice;
use jolt_equivalence::commitment_oracle::{
    append_bolt_preamble, run_generated_bolt_commitment_pair_with_cycles,
    transcript_with_bolt_commitment_trace, BoltPreambleSource, BoltTranscript,
};
use jolt_equivalence::core_oracle::{
    core_muldiv_commitment_fixture_at_log_t, CoreMuldivCommitmentFixture,
};
use jolt_prover::default_prover_programs;
use jolt_transcript::Transcript;

const TRANSCRIPT_LABEL: &[u8] = b"Jolt";

/// `BoltPreambleSource` mirroring `jolt_host::prove_program`'s preamble:
/// real fixture I/O+params but `preprocessing_digest = [0u8; 32]`.
struct ProveProgramPreambleSource<'a> {
    inner: &'a CoreMuldivCommitmentFixture,
}

impl<'a> BoltPreambleSource for ProveProgramPreambleSource<'a> {
    fn program_io(&self) -> &JoltDevice {
        self.inner.program_io()
    }
    fn preprocessing_digest(&self) -> [u8; 32] {
        [0u8; 32]
    }
    fn ram_k(&self) -> u64 {
        self.inner.ram_k()
    }
    fn trace_length(&self) -> u64 {
        self.inner.trace_length()
    }
    fn entry_address(&self) -> u64 {
        self.inner.entry_address()
    }
    fn ram_rw_phase1_num_rounds(&self) -> u64 {
        self.inner.ram_rw_phase1_num_rounds()
    }
    fn ram_rw_phase2_num_rounds(&self) -> u64 {
        self.inner.ram_rw_phase2_num_rounds()
    }
    fn registers_rw_phase1_num_rounds(&self) -> u64 {
        self.inner.registers_rw_phase1_num_rounds()
    }
    fn registers_rw_phase2_num_rounds(&self) -> u64 {
        self.inner.registers_rw_phase2_num_rounds()
    }
    fn log_k_chunk(&self) -> u64 {
        self.inner.log_k_chunk()
    }
    fn lookups_ra_virtual_log_k_chunk(&self) -> u64 {
        self.inner.lookups_ra_virtual_log_k_chunk()
    }
    fn dory_layout(&self) -> u64 {
        self.inner.dory_layout()
    }
}

#[test]
#[ignore = "diagnostic: commitment-replay vs commitment-direct stage chain parity at log_t=9"]
fn commitment_direct_vs_replay_stage_chain_parity_at_log_t_9() {
    let fixture = core_muldiv_commitment_fixture_at_log_t(9);
    let preamble = ProveProgramPreambleSource { inner: &fixture };
    let programs = default_prover_programs();
    let r1cs_key = fixture.r1cs_key();
    let stage1_data = fixture.stage1_outer_rv64_data(&r1cs_key);
    let ram_data = fixture.stage2_ram_data();

    // ===== Variant A: REPLAY =====
    // Run commitment phase against a fresh transcript, then build the
    // prover transcript by preamble + replayed commitment-trace bytes.
    let (a_commit_trace, _) = run_generated_bolt_commitment_pair_with_cycles(
        programs.commitment,
        jolt_verifier::default_verifier_programs().commitment,
        &fixture.pcs_setup,
        &fixture.cycle_inputs,
    );
    let mut a_transcript = transcript_with_bolt_commitment_trace(&preamble, &a_commit_trace);

    // ===== Variant B: DIRECT =====
    // Build preamble-loaded transcript, then call
    // `prove_commitment_phase_with_program` directly against it — exactly
    // what `jolt_host::prove_program` does.
    let mut b_transcript = BoltTranscript::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut b_transcript, &preamble);
    let commitment_sources = jolt_witness::commitment_trace_sources(&fixture.cycle_inputs);
    let oracle_inputs = jolt_prover::stages::commitment::CommitmentOracleInputs::from_trace_sources(
        &commitment_sources,
        None,
        None,
    );
    let mut commitment_inputs =
        jolt_prover::stages::commitment::SparseCommitmentInputs::new(oracle_inputs);
    let _b_commitment_artifacts =
        jolt_prover::stages::commitment::prove_commitment_phase_with_program(
            programs.commitment,
            &mut commitment_inputs,
            &fixture.pcs_setup,
            &mut b_transcript,
        )
        .expect("Variant B commitment phase");

    // ----- Stage 1 -----
    let a_stage1 = jolt_prover::prove_stage1_outer_with_witness_inputs(
        programs.stage1_outer,
        r1cs_key.num_cycle_vars(),
        &stage1_data,
        &mut a_transcript,
    )
    .expect("Variant A Stage 1");
    let b_stage1 = jolt_prover::prove_stage1_outer_with_witness_inputs(
        programs.stage1_outer,
        r1cs_key.num_cycle_vars(),
        &stage1_data,
        &mut b_transcript,
    )
    .expect("Variant B Stage 1");

    // ----- Stage 2 -----
    let a_stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(programs.stage2, &a_stage1)
            .expect("Variant A Stage 2 openings");
    let a_stage2 = jolt_prover::prove_stage2_with_witness_inputs(
        programs.stage2,
        &a_stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut a_transcript,
    )
    .expect("Variant A Stage 2");
    let b_stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(programs.stage2, &b_stage1)
            .expect("Variant B Stage 2 openings");
    let b_stage2 = jolt_prover::prove_stage2_with_witness_inputs(
        programs.stage2,
        &b_stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut b_transcript,
    )
    .expect("Variant B Stage 2");

    // ----- Stage 3 -----
    let a_stage3_openings =
        jolt_prover::stage3_opening_inputs_from_artifacts(programs.stage3, &a_stage1, &a_stage2)
            .expect("Variant A Stage 3 openings");
    let a_stage3 = jolt_prover::prove_stage3_with_witness_inputs(
        programs.stage3,
        &a_stage3_openings,
        &fixture.stage3_cycles,
        &mut a_transcript,
    )
    .expect("Variant A Stage 3");
    let b_stage3_openings =
        jolt_prover::stage3_opening_inputs_from_artifacts(programs.stage3, &b_stage1, &b_stage2)
            .expect("Variant B Stage 3 openings");
    let b_stage3 = jolt_prover::prove_stage3_with_witness_inputs(
        programs.stage3,
        &b_stage3_openings,
        &fixture.stage3_cycles,
        &mut b_transcript,
    )
    .expect("Variant B Stage 3");

    // ----- Stage 4 -----
    let a_stage4_openings = jolt_prover::stage4_opening_inputs_from_artifacts(
        programs.stage4,
        &fixture.initial_ram_state,
        &a_stage2,
        &a_stage3,
    )
    .expect("Variant A Stage 4 openings");
    let a_stage4 = jolt_prover::prove_stage4_with_trace_witness_inputs(
        programs.stage4,
        &a_stage4_openings,
        1 << fixture.params.register_log_k,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut a_transcript,
    )
    .expect("Variant A Stage 4");
    let b_stage4_openings = jolt_prover::stage4_opening_inputs_from_artifacts(
        programs.stage4,
        &fixture.initial_ram_state,
        &b_stage2,
        &b_stage3,
    )
    .expect("Variant B Stage 4 openings");
    let b_stage4 = jolt_prover::prove_stage4_with_trace_witness_inputs(
        programs.stage4,
        &b_stage4_openings,
        1 << fixture.params.register_log_k,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut b_transcript,
    )
    .expect("Variant B Stage 4");

    // ----- Stage 5 -----
    let a_stage5_openings =
        jolt_prover::stage5_opening_inputs_from_artifacts(programs.stage5, &a_stage2, &a_stage4)
            .expect("Variant A Stage 5 openings");
    let a_stage5 = jolt_prover::prove_stage5_with_trace_witness_inputs(
        programs.stage5,
        &a_stage5_openings,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        1 << fixture.params.register_log_k,
        &fixture.stage5_lookup_indices,
        &fixture.stage5_lookup_table_indices,
        &fixture.stage5_is_interleaved_operands,
        fixture.params.lookups_ra_virtual_log_k_chunk,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut a_transcript,
    )
    .expect("Variant A Stage 5");
    let b_stage5_openings =
        jolt_prover::stage5_opening_inputs_from_artifacts(programs.stage5, &b_stage2, &b_stage4)
            .expect("Variant B Stage 5 openings");
    let b_stage5 = jolt_prover::prove_stage5_with_trace_witness_inputs(
        programs.stage5,
        &b_stage5_openings,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        1 << fixture.params.register_log_k,
        &fixture.stage5_lookup_indices,
        &fixture.stage5_lookup_table_indices,
        &fixture.stage5_is_interleaved_operands,
        fixture.params.lookups_ra_virtual_log_k_chunk,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut b_transcript,
    )
    .expect("Variant B Stage 5");

    // Helper: compare artifacts at each stage. Panics on first diff.
    macro_rules! cmp_stage {
        ($label:literal, $a:expr, $b:expr) => {{
            let a_sums = &$a.sumchecks;
            let b_sums = &$b.sumchecks;
            assert_eq!(
                a_sums.len(),
                b_sums.len(),
                "{}: sumcheck count differs (replay={}, direct={})",
                $label,
                a_sums.len(),
                b_sums.len()
            );
            for (i, (a, b)) in a_sums.iter().zip(b_sums.iter()).enumerate() {
                assert_eq!(
                    a.driver, b.driver,
                    "{} sumcheck[{}] driver name differs",
                    $label, i
                );
                if a.point != b.point {
                    panic!(
                        "{} sumcheck[{}] driver={} POINT differs (len {} vs {})\n  replay: {:?}\n  direct: {:?}",
                        $label, i, a.driver, a.point.len(), b.point.len(), a.point, b.point
                    );
                }
                assert_eq!(
                    a.evals.len(),
                    b.evals.len(),
                    "{} sumcheck[{}] driver={} evals count differs",
                    $label, i, a.driver
                );
                for (ea, eb) in a.evals.iter().zip(b.evals.iter()) {
                    assert_eq!(ea.name, eb.name, "{} sumcheck[{}] eval name differs", $label, i);
                    if ea.value != eb.value {
                        panic!(
                            "{} sumcheck[{}] driver={} eval '{}' differs:\n  replay: {:?}\n  direct: {:?}",
                            $label, i, a.driver, ea.name, ea.value, eb.value
                        );
                    }
                }
                let a_polys = &a.proof.round_polynomials;
                let b_polys = &b.proof.round_polynomials;
                assert_eq!(
                    a_polys.len(),
                    b_polys.len(),
                    "{} sumcheck[{}] driver={} round_polynomials count differs",
                    $label, i, a.driver
                );
                for (r, (pa, pb)) in a_polys.iter().zip(b_polys.iter()).enumerate() {
                    let ca = pa.clone().into_coefficients();
                    let cb = pb.clone().into_coefficients();
                    if ca != cb {
                        panic!(
                            "{} sumcheck[{}] driver={} round_polynomial[{}] differs:\n  replay: {:?}\n  direct: {:?}",
                            $label, i, a.driver, r, ca, cb
                        );
                    }
                }
            }
            eprintln!("[direct-vs-replay] {}: artifacts MATCH", $label);
        }};
    }

    cmp_stage!("Stage 1", a_stage1, b_stage1);
    cmp_stage!("Stage 2", a_stage2, b_stage2);
    cmp_stage!("Stage 3", a_stage3, b_stage3);
    cmp_stage!("Stage 4", a_stage4, b_stage4);
    cmp_stage!("Stage 5", a_stage5, b_stage5);

    // ----- Stage 6 -----
    let a_stage6_openings = jolt_prover::stage6_opening_inputs_from_artifacts(
        programs.stage6,
        &a_stage1,
        &a_stage2,
        &a_stage3,
        &a_stage4,
        &a_stage5,
    )
    .expect("Variant A Stage 6 openings");
    let a_stage6_bytecode = jolt_prover::stage6_bytecode_read_raf_data_from_witness_entries(
        &fixture.stage6_bytecode_entries,
        fixture.stage6_entry_bytecode_index,
        fixture.stage6_num_lookup_tables,
    );
    let a_stage6 = jolt_prover::prove_stage6_with_trace_witness_inputs(
        programs.stage6,
        &a_stage6_openings,
        a_stage6_bytecode.as_input(),
        fixture.stage6_witness_params(),
        &fixture.cycle_inputs,
        fixture.params.instruction_ra_virtual_d,
        &mut a_transcript,
    );

    let b_stage6_openings = jolt_prover::stage6_opening_inputs_from_artifacts(
        programs.stage6,
        &b_stage1,
        &b_stage2,
        &b_stage3,
        &b_stage4,
        &b_stage5,
    )
    .expect("Variant B Stage 6 openings");
    let b_stage6_bytecode = jolt_prover::stage6_bytecode_read_raf_data_from_witness_entries(
        &fixture.stage6_bytecode_entries,
        fixture.stage6_entry_bytecode_index,
        fixture.stage6_num_lookup_tables,
    );
    let b_stage6 = jolt_prover::prove_stage6_with_trace_witness_inputs(
        programs.stage6,
        &b_stage6_openings,
        b_stage6_bytecode.as_input(),
        fixture.stage6_witness_params(),
        &fixture.cycle_inputs,
        fixture.params.instruction_ra_virtual_d,
        &mut b_transcript,
    );

    match (a_stage6, b_stage6) {
        (Ok(a), Ok(b)) => {
            cmp_stage!("Stage 6", a, b);
            eprintln!(
                "[direct-vs-replay] BOTH variants succeed through Stage 6 — \
                 the commitment-style (replay vs direct) is NOT the cause of \
                 `prove_program`'s Stage 6 failure. Bug must lie in live-trace \
                 data construction vs fixture (cycle_inputs, padding, bytecode, \
                 ram_K, memory_layout, etc.)."
            );
        }
        (Err(ae), Ok(_)) => panic!(
            "Stage 6: Variant A (replay) FAILED while Variant B (direct) SUCCEEDED: {ae:?}\n\
             Unexpected — both use the same fixture and programs."
        ),
        (Ok(_), Err(be)) => panic!(
            "Stage 6: Variant A (replay) SUCCEEDED while Variant B (direct) FAILED: {be:?}\n\
             ===> First-divergence localized to the commitment-style structural \
             difference. The replay path and the direct-call path produce \
             different Stage 6 transcripts even with identical fixture+programs.\n\
             This is THE bug behind `jolt_host::prove_program`. Switching \
             `prove_program` to record-then-replay (matching bolt_oracle) fixes it."
        ),
        (Err(ae), Err(be)) => panic!(
            "Stage 6: BOTH variants failed.\n  replay: {ae:?}\n  direct: {be:?}"
        ),
    }
}
