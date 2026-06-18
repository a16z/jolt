#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_muldiv_core_proof_is_accepted() {
    support::assert_accepts(crate::support::core_fixtures::standard_muldiv_case().verify());
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_muldiv_proof_canonical_bytes_roundtrip() {
    crate::support::core_fixtures::standard_muldiv_case().assert_canonical_bytes_roundtrip();
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_fibonacci_small_core_proof_is_accepted() {
    support::assert_accepts(
        crate::support::core_fixtures::standard_fibonacci_small_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_fibonacci_medium_core_proof_is_accepted() {
    support::assert_accepts(
        crate::support::core_fixtures::standard_fibonacci_medium_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_memory_ops_core_proof_is_accepted() {
    support::assert_accepts(crate::support::core_fixtures::standard_memory_ops_case().verify());
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_collatz_small_core_proof_is_accepted() {
    support::assert_accepts(crate::support::core_fixtures::standard_collatz_small_case().verify());
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[ignore = "hash-heavy fixture should use serialized artifacts before it is active by default"]
fn standard_sha2_small_core_proof_is_accepted() {
    support::assert_accepts(crate::support::core_fixtures::standard_sha2_small_case().verify());
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn standard_committed_muldiv_core_proof_is_accepted() {
    support::assert_accepts(
        crate::support::core_fixtures::standard_committed_muldiv_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate and cast this core fixture"]
fn standard_muldiv_core_proof_is_accepted() {}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to load or live-generate diversified core fixtures"]
fn diversified_standard_core_proofs_are_accepted() {}
