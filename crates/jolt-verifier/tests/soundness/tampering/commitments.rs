#![cfg_attr(
    all(feature = "prover-fixtures", not(feature = "zk")),
    expect(
        clippy::expect_used,
        reason = "fixture helpers should fail loudly when stored verifier NARG is malformed"
    )
)]

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::{
    narg_frame_ranges, replace_narg_frame_body, tamper_manifest, verifier_fixtures,
};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_dory::DoryCommitment;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tampered_commitment_order_reject() {
    let base = verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.order"),
        &base,
        |case| {
            let mut commitments = proof_commitments_from_narg(case);
            commitments.swap(0, 1);
            write_proof_commitments_to_narg(case, &commitments);
        },
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tampered_commitment_value_reject() {
    let base = verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.value"),
        &base,
        |case| {
            let mut commitments = proof_commitments_from_narg(case);
            commitments[0] = commitments[1].clone();
            write_proof_commitments_to_narg(case, &commitments);
        },
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tampered_commitment_count_reject() {
    let base = verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.missing"),
        &base,
        |case| {
            let body = commitment_frame_body(case);
            let _ = case.proof.narg.remove(body.start);
            let new_len = body.len() - 1;
            case.proof.narg[0..8].copy_from_slice(&(new_len as u64).to_le_bytes());
        },
    );
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.extra"),
        &base,
        |case| {
            let body = commitment_frame_body(case);
            case.proof.narg.insert(body.end, 0xff);
            let new_len = body.len() + 1;
            case.proof.narg[0..8].copy_from_slice(&(new_len as u64).to_le_bytes());
        },
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tampered_joint_opening_proof_reject() {
    let base = verifier_fixtures::standard_muldiv_case();
    let replacement = verifier_fixtures::standard_fibonacci_small_case()
        .proof
        .joint_opening_proof;
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.joint_opening_proof"),
        &base,
        |case| {
            case.proof.joint_opening_proof = replacement;
        },
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "direct commitment tampering fixtures are not wired yet"]
fn tampered_commitment_order_reject() {}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn proof_commitments_from_narg(
    case: &verifier_fixtures::VerifierFixtureCase,
) -> Vec<DoryCommitment> {
    let body = commitment_frame_body(case);
    jolt_transcript::deserialize_slice(&case.proof.narg[body])
        .expect("fixture commitment NARG frame should decode")
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn write_proof_commitments_to_narg(
    case: &mut verifier_fixtures::VerifierFixtureCase,
    commitments: &[DoryCommitment],
) {
    replace_narg_frame_body(
        &mut case.proof.narg,
        0,
        jolt_transcript::serialize_slice(commitments),
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn commitment_frame_body(case: &verifier_fixtures::VerifierFixtureCase) -> std::ops::Range<usize> {
    let ranges = narg_frame_ranges(&case.proof.narg);
    ranges
        .first()
        .expect("fixture should have a proof commitment NARG frame")
        .body
        .clone()
}
