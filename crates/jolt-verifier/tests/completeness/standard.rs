#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    completeness::cases,
    support::{completeness_expectation, HarnessExpectation},
};
use crate::{support, support::dory_pedersen};

#[test]
fn valid_standard_dory_pedersen_proof_reaches_current_frontier() {
    support::assert_reaches_current_frontier(dory_pedersen::standard_case().verify());
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_muldiv_core_proof_reaches_frontier() {
    support::assert_reaches_current_frontier(
        crate::support::core_fixtures::standard_muldiv_case().verify(),
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
