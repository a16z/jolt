#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn verifier_rejects_public_io_mismatch() {
    support::assert_rejects(
        crate::support::verifier_fixtures::public_io_memory_layout_mismatch_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to run this verifier-transitivity fixture"]
fn verifier_rejects_public_io_mismatch() {}
