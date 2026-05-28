#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::core_fixtures;

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn precompat_dory_commitment_bridge_round_trips_nontrivial_commitment() {
    let case = core_fixtures::standard_muldiv_precompat_case();

    assert!(case.first_dory_commitment_round_trips_through_modular_type());
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to check compat commitment conversion"]
fn precompat_dory_commitment_bridge_round_trips_nontrivial_commitment() {}
