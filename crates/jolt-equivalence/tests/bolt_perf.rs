//! Dedicated perf-oracle gates for real-data core-vs-Bolt runs.

use jolt_equivalence::bolt_oracle::assert_bolt_full_real_trace_self_parity;
use jolt_equivalence::core_oracle::core_sha2_chain_commitment_fixture;
use jolt_equivalence::perf::maybe_setup_perf_trace;
use jolt_inlines_sha2 as _;

#[test]
#[ignore = "run by the Bolt perf-oracle CI workflow"]
fn bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle() {
    maybe_setup_perf_trace("bolt_sha2_chain_2_16_core_vs_bolt");
    assert_bolt_full_real_trace_self_parity(core_sha2_chain_commitment_fixture(16), true);
}

#[test]
#[ignore = "run by the Bolt perf-oracle CI workflow"]
fn bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle() {
    maybe_setup_perf_trace("bolt_sha2_chain_2_20_core_vs_bolt");
    assert_bolt_full_real_trace_self_parity(core_sha2_chain_commitment_fixture(20), true);
}
