#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_muldiv_verifier_proof_is_accepted() {
    support::assert_accepts(crate::support::verifier_fixtures::standard_muldiv_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_fibonacci_small_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_fibonacci_small_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_fibonacci_medium_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_fibonacci_medium_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_memory_ops_verifier_proof_is_accepted() {
    support::assert_accepts(crate::support::verifier_fixtures::standard_memory_ops_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_collatz_small_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_collatz_small_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[ignore = "hash-heavy fixture should use serialized fixtures before it is active by default"]
fn standard_sha2_small_verifier_proof_is_accepted() {
    support::assert_accepts(crate::support::verifier_fixtures::standard_sha2_small_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_committed_muldiv_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_committed_muldiv_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate this verifier fixture"]
fn standard_muldiv_verifier_proof_is_accepted() {}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to load or live-generate diversified verifier fixtures"]
fn diversified_standard_verifier_objects_are_accepted() {}
