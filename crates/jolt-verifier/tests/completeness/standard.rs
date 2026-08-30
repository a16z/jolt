#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

const DORY_MULDIV_PROOF_DIGEST: [u8; 32] = [
    74, 157, 108, 227, 221, 184, 160, 181, 189, 204, 163, 209, 56, 155, 205, 46, 148, 247, 96, 38,
    255, 115, 71, 154, 146, 236, 110, 243, 111, 66, 219, 63,
];
const DORY_COMMITTED_MULDIV_PROOF_DIGEST: [u8; 32] = [
    113, 41, 118, 10, 141, 181, 219, 89, 177, 44, 194, 60, 154, 251, 167, 61, 115, 217, 19, 176,
    253, 114, 12, 224, 64, 51, 80, 197, 52, 33, 37, 144,
];

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_muldiv_verifier_proof_is_accepted() {
    let case = crate::support::verifier_fixtures::standard_muldiv_case();
    support::assert_accepts(case.verify());
    assert_eq!(
        support::proof_wire_digest(&case.proof),
        DORY_MULDIV_PROOF_DIGEST
    );
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
    let case = crate::support::verifier_fixtures::standard_committed_muldiv_case();
    support::assert_accepts(case.verify());
    assert_eq!(
        support::proof_wire_digest(&case.proof),
        DORY_COMMITTED_MULDIV_PROOF_DIGEST
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_address_major_verifier_proofs_are_accepted() {
    for case in [
        crate::support::verifier_fixtures::fresh_standard_muldiv_address_major_case(),
        crate::support::verifier_fixtures::fresh_standard_committed_muldiv_address_major_case(2),
        crate::support::verifier_fixtures::fresh_standard_committed_muldiv_address_major_case(64),
    ] {
        support::assert_accepts(case.verify());
    }
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate this verifier fixture"]
fn standard_muldiv_verifier_proof_is_accepted() {}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to load or live-generate diversified verifier fixtures"]
fn diversified_standard_verifier_objects_are_accepted() {}
