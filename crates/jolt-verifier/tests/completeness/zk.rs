use crate::{
    completeness::cases,
    support::{self, completeness_expectation, dory_pedersen, HarnessExpectation},
};

#[test]
fn valid_zk_dory_pedersen_proof_reaches_current_frontier() {
    support::assert_zk_reaches_current_frontier(dory_pedersen::zk_case().verify());
}

#[test]
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[ignore = "ZK core fixture is wired, but enabling it is deferred until the ZK verifier frontier"]
fn zk_muldiv_core_proof_reaches_frontier() {
    support::assert_zk_reaches_current_frontier(
        crate::support::core_fixtures::zk_muldiv_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), not(feature = "zk")))]
#[ignore = "enable --features core-fixtures,zk to live-generate and cast this core ZK fixture"]
fn zk_muldiv_core_proof_reaches_frontier() {
    assert_eq!(
        completeness_expectation(cases::ZK_MULDIV_SMALL),
        HarnessExpectation::FutureCheckpoint,
    );
}

#[test]
#[ignore = "prefix BlindFold fixture generation is not wired yet"]
fn zk_stage1_prefix_reaches_frontier() {
    assert_eq!(
        completeness_expectation(cases::ZK_STAGE1_PREFIX),
        HarnessExpectation::FutureCheckpoint,
    );
}
