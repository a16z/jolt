#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support::proof_claims::attach_empty_opening_claims;
#[cfg(feature = "prover-fixtures")]
use crate::support::tamper_manifest;
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_verifier::proof::JoltProofClaims;

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn zk_claim_payload_in_clear_mode_rejects_now() {
    let base = verifier_fixture_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.claims.mode_payload"),
        &base,
        |case| {
            case.proof.claims = JoltProofClaims::Zk;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn unexpected_zk_opening_claims_reject_now() {
    assert_zk_target_active("proof.claims.mode_payload");
    let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
    attach_empty_opening_claims(&mut case.proof);

    support::assert_zk_rejects(case.verify());
}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures to live-generate and tamper verifier-native proofs"]
fn tampered_mixed_proof_shape_reject() {}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn verifier_fixture_case() -> crate::support::verifier_fixtures::VerifierFixtureCase {
    crate::support::verifier_fixtures::standard_muldiv_case()
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn assert_zk_target_active(name: &str) {
    let target = tamper_manifest::required_target(name);
    tamper_manifest::assert_manifest_target_is_active(target);
    assert!(
        target.mode.includes(true),
        "tamper target mode does not include ZK: {target:?}"
    );
}
