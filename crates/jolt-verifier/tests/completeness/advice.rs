#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_advice_consumer_core_proof_is_accepted() {
    support::assert_accepts(
        crate::support::core_fixtures::standard_advice_consumer_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate and cast this advice fixture"]
fn standard_advice_consumer_core_proof_is_accepted() {}
