#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support;
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    completeness::cases,
    support::{completeness_expectation, HarnessExpectation},
};

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_advice_consumer_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_advice_consumer_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate and cast this advice fixture"]
fn standard_advice_consumer_core_proof_reaches_frontier() {
    assert_eq!(
        completeness_expectation(cases::STANDARD_ADVICE_CONSUMER),
        HarnessExpectation::ReachesFrontier,
    );
}
