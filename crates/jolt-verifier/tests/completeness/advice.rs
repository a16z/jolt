#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_advice_consumer_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_advice_consumer_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate this advice fixture"]
fn standard_advice_consumer_verifier_proof_is_accepted() {}
