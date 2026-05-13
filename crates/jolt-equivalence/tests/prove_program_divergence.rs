//! DIAGNOSTIC: find first stage where bolt_oracle (fresh bolt programs) and
//! prove_program-style (`default_prover_programs()`) artifacts diverge when
//! fed identical fixture data at log_t=9.
//!
//! Hypothesis: the goldens-committed `default_prover_programs()` plans differ
//! from the freshly-emitted bolt plans, causing prove_program's Stage 6
//! round_poly check to fail while bolt_oracle's path passes.
//!
//! This test fails LOUDLY at the first stage where prover artifacts differ
//! between the two program-plan sources. If all stages 1..5 match, the
//! divergence is isolated to Stage 6.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::collapsible_if,
    clippy::too_many_lines,
    clippy::needless_lifetimes,
    unfulfilled_lint_expectations,
    reason = "diagnostic test fails fast on internal mismatches"
)]

use common::jolt_device::JoltDevice;
use jolt_equivalence::bolt_programs::{
    bolt_commitment_programs_with_params, bolt_stage1_programs_with_params,
    bolt_stage2_programs_with_params, bolt_stage3_programs_with_params,
    bolt_stage4_programs_with_params, bolt_stage5_programs_with_params,
    bolt_stage6_programs_with_params,
};
use jolt_equivalence::commitment_oracle::{
    run_generated_bolt_commitment_pair_with_cycles, transcript_with_bolt_commitment_trace,
    BoltPreambleSource,
};
use jolt_equivalence::core_oracle::{
    core_muldiv_commitment_fixture_at_log_t, CoreMuldivCommitmentFixture,
};
use jolt_equivalence::plan_adapters::{
    leak_generated_commitment_prover_program, leak_generated_commitment_verifier_program,
    leak_stage1_program, leak_stage2_program, leak_stage3_program, leak_stage4_program,
    leak_stage5_program, leak_stage6_program,
};
use jolt_prover::default_prover_programs;

/// `BoltPreambleSource` matching `jolt_host::prove_program`'s preamble:
/// real fixture I/O+params, but `preprocessing_digest = [0u8; 32]` as
/// `prove_program` currently writes.
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
#[ignore = "diagnostic: find first stage where bolt_oracle vs default_prover_programs diverge"]
fn prove_program_vs_bolt_oracle_artifact_divergence_at_log_t_9() {
    let fixture = core_muldiv_commitment_fixture_at_log_t(9);

    // Path A: bolt_oracle uses fresh bolt programs.
    let (a_commitment_prover, a_commitment_verifier) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (a_stage1_prover, _) = bolt_stage1_programs_with_params(&fixture.params);
    let (a_stage2_prover, _) = bolt_stage2_programs_with_params(&fixture.params);
    let (a_stage3_prover, _) = bolt_stage3_programs_with_params(&fixture.params);
    let (a_stage4_prover, _) = bolt_stage4_programs_with_params(&fixture.params);
    let (a_stage5_prover, _) = bolt_stage5_programs_with_params(&fixture.params);
    let (a_stage6_prover, _) = bolt_stage6_programs_with_params(&fixture.params);
    let a_commitment_prover_plan = leak_generated_commitment_prover_program(&a_commitment_prover);
    let a_commitment_verifier_plan =
        leak_generated_commitment_verifier_program(&a_commitment_verifier);
    let a_stage1_plan = leak_stage1_program(&a_stage1_prover);
    let a_stage2_plan = leak_stage2_program(&a_stage2_prover);
    let a_stage3_plan = leak_stage3_program(&a_stage3_prover);
    let a_stage4_plan = leak_stage4_program(&a_stage4_prover);
    let a_stage5_plan = leak_stage5_program(&a_stage5_prover);
    let a_stage6_plan = leak_stage6_program(&a_stage6_prover);

    // Path B: prove_program uses committed goldens.
    let b_programs = default_prover_programs();
    let b_verifier_programs = jolt_verifier::default_verifier_programs();

    // ===== Path A =====
    let (a_commitment_prover_trace, _) = run_generated_bolt_commitment_pair_with_cycles(
        a_commitment_prover_plan,
        a_commitment_verifier_plan,
        &fixture.pcs_setup,
        &fixture.cycle_inputs,
    );

    let r1cs_key = fixture.r1cs_key();
    let stage1_data = fixture.stage1_outer_rv64_data(&r1cs_key);
    let ram_data = fixture.stage2_ram_data();

    // Detect commitment-plan divergence: if the goldens commitment plan
    // produces different Dory commitments than the fresh bolt commitment
    // plan on identical cycle inputs, every downstream transcript challenge
    // will diverge.
    let (b_commitment_prover_trace_diag, _) = run_generated_bolt_commitment_pair_with_cycles(
        b_programs.commitment,
        b_verifier_programs.commitment,
        &fixture.pcs_setup,
        &fixture.cycle_inputs,
    );
    let a_commit_count = a_commitment_prover_trace.commitments.len();
    let b_commit_count = b_commitment_prover_trace_diag.commitments.len();
    eprintln!(
        "[divergence-diag] Path A commitments={}, Path B commitments={}",
        a_commit_count, b_commit_count
    );
    if a_commit_count != b_commit_count {
        panic!(
            "Commitment phase: different commitment counts (bolt={}, prove_pg={}). \
             Goldens commitment plan emits a structurally different commitment set.",
            a_commit_count, b_commit_count
        );
    }
    let mut commit_diffs = Vec::new();
    for (i, (ca, cb)) in a_commitment_prover_trace
        .commitments
        .iter()
        .zip(b_commitment_prover_trace_diag.commitments.iter())
        .enumerate()
    {
        if format!("{:?}", ca) != format!("{:?}", cb) {
            commit_diffs.push(i);
        }
    }
    if !commit_diffs.is_empty() {
        panic!(
            "Commitment phase: commitments[i] differ at indices {:?} ({} of {} commits). \
             First-divergence is in the commitment phase — Stage 1+ transcripts can never match.",
            commit_diffs,
            commit_diffs.len(),
            a_commit_count
        );
    }
    eprintln!(
        "[divergence-diag] commitment phase commitments MATCH bit-for-bit ({} commits)",
        a_commit_count
    );

    let mut a_transcript =
        transcript_with_bolt_commitment_trace(&fixture, &a_commitment_prover_trace);
    let a_stage1 = jolt_prover::prove_stage1_outer_with_witness_inputs(
        a_stage1_plan,
        r1cs_key.num_cycle_vars(),
        &stage1_data,
        &mut a_transcript,
    )
    .expect("Path A Stage 1");

    let a_stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(a_stage2_plan, &a_stage1)
            .expect("Path A Stage 2 openings");
    let a_stage2 = jolt_prover::prove_stage2_with_witness_inputs(
        a_stage2_plan,
        &a_stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut a_transcript,
    )
    .expect("Path A Stage 2");

    let a_stage3_openings = jolt_prover::stage3_opening_inputs_from_artifacts(
        a_stage3_plan,
        &a_stage1,
        &a_stage2,
    )
    .expect("Path A Stage 3 openings");
    let a_stage3 = jolt_prover::prove_stage3_with_witness_inputs(
        a_stage3_plan,
        &a_stage3_openings,
        &fixture.stage3_cycles,
        &mut a_transcript,
    )
    .expect("Path A Stage 3");

    let a_stage4_openings = jolt_prover::stage4_opening_inputs_from_artifacts(
        a_stage4_plan,
        &fixture.initial_ram_state,
        &a_stage2,
        &a_stage3,
    )
    .expect("Path A Stage 4 openings");
    let a_stage4 = jolt_prover::prove_stage4_with_trace_witness_inputs(
        a_stage4_plan,
        &a_stage4_openings,
        1 << fixture.params.register_log_k,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut a_transcript,
    )
    .expect("Path A Stage 4");

    let a_stage5_openings = jolt_prover::stage5_opening_inputs_from_artifacts(
        a_stage5_plan,
        &a_stage2,
        &a_stage4,
    )
    .expect("Path A Stage 5 openings");
    let a_stage5 = jolt_prover::prove_stage5_with_trace_witness_inputs(
        a_stage5_plan,
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
    .expect("Path A Stage 5");

    // ===== Path B =====
    let b_preamble_src = ProveProgramPreambleSource { inner: &fixture };
    let b_commitment_prover_trace = b_commitment_prover_trace_diag;
    let mut b_transcript =
        transcript_with_bolt_commitment_trace(&b_preamble_src, &b_commitment_prover_trace);

    // Additionally, run a "Path A with prove_program preamble" sub-trial to
    // attribute the Stage 1 divergence: if Path A's Stage 1 first-round
    // challenge under ProveProgramPreambleSource equals Path B's, then the
    // sole cause is the preamble (preprocessing_digest=[0u8;32]). If they
    // still differ, the commitment-trace records appended to the transcript
    // diverge even though commitments match.
    {
        let mut a_with_b_preamble_transcript =
            transcript_with_bolt_commitment_trace(&b_preamble_src, &a_commitment_prover_trace);
        let a_with_b_pre = jolt_prover::prove_stage1_outer_with_witness_inputs(
            a_stage1_plan,
            r1cs_key.num_cycle_vars(),
            &stage1_data,
            &mut a_with_b_preamble_transcript,
        )
        .expect("Path A (B preamble) Stage 1");
        eprintln!(
            "[divergence-diag] sub-trial: Path A under prove_program preamble \
             stage1.uniskip.point[0] = {:?}",
            a_with_b_pre.sumchecks[0].point[0]
        );
    }
    let b_stage1 = jolt_prover::prove_stage1_outer_with_witness_inputs(
        b_programs.stage1_outer,
        r1cs_key.num_cycle_vars(),
        &stage1_data,
        &mut b_transcript,
    )
    .expect("Path B Stage 1");

    let b_stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(b_programs.stage2, &b_stage1)
            .expect("Path B Stage 2 openings");
    let b_stage2 = jolt_prover::prove_stage2_with_witness_inputs(
        b_programs.stage2,
        &b_stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut b_transcript,
    )
    .expect("Path B Stage 2");

    let b_stage3_openings = jolt_prover::stage3_opening_inputs_from_artifacts(
        b_programs.stage3,
        &b_stage1,
        &b_stage2,
    )
    .expect("Path B Stage 3 openings");
    let b_stage3 = jolt_prover::prove_stage3_with_witness_inputs(
        b_programs.stage3,
        &b_stage3_openings,
        &fixture.stage3_cycles,
        &mut b_transcript,
    )
    .expect("Path B Stage 3");

    let b_stage4_openings = jolt_prover::stage4_opening_inputs_from_artifacts(
        b_programs.stage4,
        &fixture.initial_ram_state,
        &b_stage2,
        &b_stage3,
    )
    .expect("Path B Stage 4 openings");
    let b_stage4 = jolt_prover::prove_stage4_with_trace_witness_inputs(
        b_programs.stage4,
        &b_stage4_openings,
        1 << fixture.params.register_log_k,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut b_transcript,
    )
    .expect("Path B Stage 4");

    let b_stage5_openings = jolt_prover::stage5_opening_inputs_from_artifacts(
        b_programs.stage5,
        &b_stage2,
        &b_stage4,
    )
    .expect("Path B Stage 5 openings");
    let b_stage5 = jolt_prover::prove_stage5_with_trace_witness_inputs(
        b_programs.stage5,
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
    .expect("Path B Stage 5");

    macro_rules! cmp_stage {
        ($label:literal, $a:expr, $b:expr) => {{
            let a_sums = &$a.sumchecks;
            let b_sums = &$b.sumchecks;
            assert_eq!(
                a_sums.len(),
                b_sums.len(),
                "{}: sumcheck count differs (bolt_oracle={}, prove_program={})",
                $label,
                a_sums.len(),
                b_sums.len()
            );
            for (i, (a, b)) in a_sums.iter().zip(b_sums.iter()).enumerate() {
                assert_eq!(
                    a.driver, b.driver,
                    "{} sumcheck[{}] driver name differs", $label, i
                );
                if a.point != b.point {
                    panic!(
                        "{} sumcheck[{}] driver={} POINT differs (len {} vs {})\n  bolt:    {:?}\n  prove_pg: {:?}",
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
                    assert_eq!(
                        ea.name, eb.name,
                        "{} sumcheck[{}] eval name differs", $label, i
                    );
                    if ea.value != eb.value {
                        panic!(
                            "{} sumcheck[{}] driver={} eval '{}' differs:\n  bolt:    {:?}\n  prove_pg: {:?}",
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
                            "{} sumcheck[{}] driver={} round_polynomial[{}] differs:\n  bolt:    {:?}\n  prove_pg: {:?}",
                            $label, i, a.driver, r, ca, cb
                        );
                    }
                }
            }
            eprintln!("[divergence-diag] {}: artifacts MATCH", $label);
        }};
    }

    cmp_stage!("Stage 1", a_stage1, b_stage1);
    cmp_stage!("Stage 2", a_stage2, b_stage2);
    cmp_stage!("Stage 3", a_stage3, b_stage3);
    cmp_stage!("Stage 4", a_stage4, b_stage4);
    cmp_stage!("Stage 5", a_stage5, b_stage5);

    // Stage 6: if both succeed we compare; if only Path B fails we've
    // localized the failure to Stage 6 with all earlier artifacts matching.
    let a_stage6_openings = jolt_prover::stage6_opening_inputs_from_artifacts(
        a_stage6_plan,
        &a_stage1,
        &a_stage2,
        &a_stage3,
        &a_stage4,
        &a_stage5,
    )
    .expect("Path A Stage 6 openings");
    let a_stage6_bytecode = jolt_prover::stage6_bytecode_read_raf_data_from_witness_entries(
        &fixture.stage6_bytecode_entries,
        fixture.stage6_entry_bytecode_index,
        fixture.stage6_num_lookup_tables,
    );
    let a_stage6 = jolt_prover::prove_stage6_with_trace_witness_inputs(
        a_stage6_plan,
        &a_stage6_openings,
        a_stage6_bytecode.as_input(),
        fixture.stage6_witness_params(),
        &fixture.cycle_inputs,
        fixture.params.instruction_ra_virtual_d,
        &mut a_transcript,
    );

    let b_stage6_openings = jolt_prover::stage6_opening_inputs_from_artifacts(
        b_programs.stage6,
        &b_stage1,
        &b_stage2,
        &b_stage3,
        &b_stage4,
        &b_stage5,
    )
    .expect("Path B Stage 6 openings");
    let b_stage6_bytecode = jolt_prover::stage6_bytecode_read_raf_data_from_witness_entries(
        &fixture.stage6_bytecode_entries,
        fixture.stage6_entry_bytecode_index,
        fixture.stage6_num_lookup_tables,
    );
    let b_stage6 = jolt_prover::prove_stage6_with_trace_witness_inputs(
        b_programs.stage6,
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
                "[divergence-diag] BOTH paths succeed through Stage 6 — \
                 any prove_program failure is downstream (Stage 7/8 or verifier)."
            );
        }
        (Err(ae), Ok(_)) => panic!(
            "Stage 6: Path A (bolt_oracle) FAILED while Path B (prove_program) SUCCEEDED: {ae:?}"
        ),
        (Ok(_), Err(be)) => panic!(
            "Stage 6: Path A (bolt_oracle) SUCCEEDED while Path B (prove_program) FAILED: {be:?}\n\
             ===> First-divergence localized to Stage 6 inside the prove call. \
             Stages 1-5 artifacts matched bit-for-bit. \
             Suspects: (1) goldens stage6 plan differs from fresh bolt stage6 plan; \
             (2) goldens commitment plan emitted different commitments, perturbing Stage 6 \
             transcript challenges; (3) prove_program's [0u8;32] preprocessing_digest perturbs \
             early transcript challenges. If A==B through Stage 5 sumchecks but Stage 6 \
             diverges, the goldens stage6 plan itself is the culprit."
        ),
        (Err(ae), Err(be)) => panic!(
            "Stage 6: BOTH paths failed.\n  Path A: {ae:?}\n  Path B: {be:?}"
        ),
    }
}
