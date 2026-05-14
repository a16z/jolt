//! Dedicated perf-oracle gates for real-data core-vs-Bolt runs.

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "perf oracle tests print sample progress and fail with gate context"
)]

use jolt_equivalence::bolt_oracle::assert_bolt_full_real_trace_self_parity;
use jolt_equivalence::core_oracle::core_sha2_chain_commitment_fixture;
use jolt_equivalence::perf::{
    check_sampled_core_vs_bolt_perf_gate, core_vs_bolt_perf_sample_count, maybe_setup_perf_trace,
    print_sampled_core_vs_bolt_perf_summary, CORE_VS_BOLT_PERF_THRESHOLDS,
};
use jolt_inlines_sha2 as _;

#[test]
#[ignore = "run by the Bolt perf-oracle CI workflow"]
fn bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle() {
    run_sha2_chain_perf_oracle("bolt_sha2_chain_2_16_core_vs_bolt", 16);
}

#[test]
#[ignore = "run by the Bolt perf-oracle CI workflow"]
fn bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle() {
    run_sha2_chain_perf_oracle("bolt_sha2_chain_2_20_core_vs_bolt", 20);
}

fn run_sha2_chain_perf_oracle(trace_name: &'static str, log_trace_len: usize) {
    maybe_setup_perf_trace(trace_name);
    let sample_count = core_vs_bolt_perf_sample_count();
    let mut samples = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        println!(
            "core-vs-Bolt perf oracle sample {}/{}",
            sample_index + 1,
            sample_count
        );
        samples.push(assert_bolt_full_real_trace_self_parity(
            core_sha2_chain_commitment_fixture(log_trace_len),
            false,
        ));
    }
    let report = check_sampled_core_vs_bolt_perf_gate(&samples, CORE_VS_BOLT_PERF_THRESHOLDS)
        .expect("sampled core-vs-Bolt perf oracle gate");
    print_sampled_core_vs_bolt_perf_summary(&samples, &report);
}
