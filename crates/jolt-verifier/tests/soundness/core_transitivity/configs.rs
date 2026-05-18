#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support;
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    soundness::core_transitivity,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn core_rejects_config_mismatch() {
    support::assert_rejects_at_or_before_current_frontier(
        crate::support::core_fixtures::invalid_trace_length_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn core_rejects_ram_domain_mismatch() {
    support::assert_rejects_at_or_before_current_frontier(
        crate::support::core_fixtures::invalid_ram_k_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to run this core-transitivity fixture"]
fn core_rejects_config_mismatch() {
    assert_eq!(
        soundness_expectation(core_transitivity::CONFIG_MISMATCH),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
