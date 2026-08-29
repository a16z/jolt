#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

const DORY_ADVICE_PROOF_DIGEST: [u8; 32] = [
    26, 33, 63, 217, 78, 115, 229, 128, 200, 162, 139, 87, 82, 225, 108, 190, 64, 65, 112, 208, 2,
    182, 123, 101, 49, 135, 34, 161, 25, 172, 31, 171,
];

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_advice_consumer_verifier_proof_is_accepted() {
    let case = crate::support::verifier_fixtures::standard_advice_consumer_case();
    support::assert_accepts(case.verify());
    assert_eq!(
        support::proof_wire_digest(&case.proof),
        DORY_ADVICE_PROOF_DIGEST
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate this advice fixture"]
fn standard_advice_consumer_verifier_proof_is_accepted() {}
