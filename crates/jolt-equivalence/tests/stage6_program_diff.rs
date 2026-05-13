//! Diagnostic: compare the committed `STAGE6_PROGRAM` golden against a
//! freshly built Bolt Stage 6 prover program at fixture parameters.
//!
//! Motivation: a Stage 6 "input claim mismatch" reproduces in
//! `jolt_host::prove_program` (which consumes `STAGE6_PROGRAM`) but NOT in
//! `bolt_oracle` (which consumes a freshly emitted program). If the goldens
//! and the fresh emission diverge, that explains the asymmetry.

#![expect(
    clippy::expect_used,
    reason = "diagnostic test; fail fast if setup fails"
)]

use bolt::protocols::jolt::JoltProtocolParams;
use jolt_equivalence::bolt_programs::bolt_stage6_programs_with_params;
use jolt_equivalence::plan_adapters::leak_stage6_program;
use jolt_kernels::stage6::Stage6CpuProgramPlan;
use jolt_prover::stages::stage6::STAGE6_PROGRAM;

fn diff_slice<T: PartialEq + std::fmt::Debug>(
    name: &str,
    fresh: &[T],
    golden: &[T],
    diffs: &mut Vec<String>,
) {
    if fresh.len() != golden.len() {
        diffs.push(format!(
            "{name}: length mismatch — fresh={} golden={}",
            fresh.len(),
            golden.len()
        ));
        let common = fresh.len().min(golden.len());
        for i in 0..common {
            if fresh[i] != golden[i] {
                diffs.push(format!(
                    "  {name}[{i}] differs:\n    fresh:  {:?}\n    golden: {:?}",
                    fresh[i], golden[i]
                ));
            }
        }
        for i in common..fresh.len() {
            diffs.push(format!("  {name}[{i}] only in fresh:  {:?}", fresh[i]));
        }
        for i in common..golden.len() {
            diffs.push(format!("  {name}[{i}] only in golden: {:?}", golden[i]));
        }
        return;
    }
    for (i, (f, g)) in fresh.iter().zip(golden.iter()).enumerate() {
        if f != g {
            diffs.push(format!(
                "{name}[{i}] differs:\n    fresh:  {f:?}\n    golden: {g:?}"
            ));
        }
    }
}

fn diff_programs(fresh: &Stage6CpuProgramPlan, golden: &Stage6CpuProgramPlan) -> Vec<String> {
    let mut diffs = Vec::new();

    if fresh.role != golden.role {
        diffs.push(format!(
            "role differs: fresh={:?} golden={:?}",
            fresh.role, golden.role
        ));
    }
    if fresh.params != golden.params {
        diffs.push(format!(
            "params differ:\n    fresh:  {:?}\n    golden: {:?}",
            fresh.params, golden.params
        ));
    }

    diff_slice("steps", fresh.steps, golden.steps, &mut diffs);
    diff_slice(
        "transcript_squeezes",
        fresh.transcript_squeezes,
        golden.transcript_squeezes,
        &mut diffs,
    );
    diff_slice(
        "transcript_absorb_bytes",
        fresh.transcript_absorb_bytes,
        golden.transcript_absorb_bytes,
        &mut diffs,
    );
    diff_slice(
        "opening_inputs",
        fresh.opening_inputs,
        golden.opening_inputs,
        &mut diffs,
    );
    diff_slice(
        "field_constants",
        fresh.field_constants,
        golden.field_constants,
        &mut diffs,
    );
    diff_slice(
        "field_exprs",
        fresh.field_exprs,
        golden.field_exprs,
        &mut diffs,
    );
    diff_slice("kernels", fresh.kernels, golden.kernels, &mut diffs);
    diff_slice("claims", fresh.claims, golden.claims, &mut diffs);
    diff_slice("batches", fresh.batches, golden.batches, &mut diffs);
    diff_slice("drivers", fresh.drivers, golden.drivers, &mut diffs);
    diff_slice(
        "instance_results",
        fresh.instance_results,
        golden.instance_results,
        &mut diffs,
    );
    diff_slice("evals", fresh.evals, golden.evals, &mut diffs);
    diff_slice(
        "point_zeros",
        fresh.point_zeros,
        golden.point_zeros,
        &mut diffs,
    );
    diff_slice(
        "point_slices",
        fresh.point_slices,
        golden.point_slices,
        &mut diffs,
    );
    diff_slice(
        "point_concats",
        fresh.point_concats,
        golden.point_concats,
        &mut diffs,
    );
    diff_slice(
        "opening_claims",
        fresh.opening_claims,
        golden.opening_claims,
        &mut diffs,
    );
    diff_slice(
        "opening_equalities",
        fresh.opening_equalities,
        golden.opening_equalities,
        &mut diffs,
    );
    diff_slice(
        "opening_batches",
        fresh.opening_batches,
        golden.opening_batches,
        &mut diffs,
    );

    diffs
}

#[test]
fn stage6_golden_matches_fresh_bolt_emission_at_fixture() {
    let params = JoltProtocolParams::fixture();
    let (fresh_prover_program, _fresh_verifier_program) =
        bolt_stage6_programs_with_params(&params);
    let fresh: &'static Stage6CpuProgramPlan = leak_stage6_program(&fresh_prover_program);
    let golden: &'static Stage6CpuProgramPlan = &STAGE6_PROGRAM;

    let diffs = diff_programs(fresh, golden);

    if diffs.is_empty() {
        eprintln!(
            "STAGE6_PROGRAM and fresh Bolt Stage6 emission are IDENTICAL at JoltProtocolParams::fixture() (log_t=9, log_k_bytecode=13, log_k_ram=13)."
        );
        return;
    }

    eprintln!(
        "Stage6CpuProgramPlan divergences (fresh Bolt emit vs jolt_prover::STAGE6_PROGRAM golden):"
    );
    eprintln!("  total field-level divergences: {}", diffs.len());
    for (i, d) in diffs.iter().enumerate() {
        eprintln!("[{i:04}] {d}");
    }
    panic!(
        "STAGE6_PROGRAM diverges from fresh Bolt emission at fixture params — {} divergence(s); see test output",
        diffs.len()
    );
}
