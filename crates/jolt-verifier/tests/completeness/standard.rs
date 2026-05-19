#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support;
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    completeness::cases,
    support::{completeness_expectation, HarnessExpectation},
};

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_muldiv_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_muldiv_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_fibonacci_small_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_fibonacci_small_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_fibonacci_medium_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_fibonacci_medium_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_memory_ops_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_memory_ops_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_collatz_small_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_collatz_small_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[ignore = "hash-heavy fixture should use serialized artifacts before it is active by default"]
fn standard_sha2_small_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_sha2_small_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate and cast this core fixture"]
fn standard_muldiv_core_proof_reaches_frontier() {
    assert_eq!(
        completeness_expectation(cases::STANDARD_MULDIV_SMALL),
        HarnessExpectation::ReachesFrontier,
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to load or live-generate diversified core fixtures"]
fn diversified_standard_core_proofs_reach_frontier() {
    for case in [
        cases::STANDARD_FIBONACCI_SMALL,
        cases::STANDARD_FIBONACCI_MEDIUM,
        cases::STANDARD_MEMORY_OPS,
        cases::STANDARD_COLLATZ_SMALL,
        cases::STANDARD_SHA2_SMALL,
    ] {
        assert_eq!(
            completeness_expectation(case),
            HarnessExpectation::ReachesFrontier,
        );
    }
}
