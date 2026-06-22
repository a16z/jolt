#![expect(
    clippy::panic,
    reason = "tamper tests fail loudly when fixture assumptions are violated"
)]

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::{core_fixtures, tamper_manifest};

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tampered_commitment_order_reject() {
    let base = core_fixtures::standard_muldiv_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.order"),
        &base,
        |case| {
            let jolt_verifier::CommitmentPayload::Dory(commitments) = &mut case.proof.commitments
            else {
                panic!("core fixture should carry Dory commitments");
            };
            std::mem::swap(&mut commitments.rd_inc, &mut commitments.ram_inc);
        },
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tampered_commitment_value_reject() {
    let base = core_fixtures::standard_muldiv_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.value"),
        &base,
        |case| {
            let jolt_verifier::CommitmentPayload::Dory(commitments) = &mut case.proof.commitments
            else {
                panic!("core fixture should carry Dory commitments");
            };
            commitments.rd_inc = commitments.ram_inc.clone();
        },
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tampered_commitment_count_reject() {
    let base = core_fixtures::standard_muldiv_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.missing"),
        &base,
        |case| {
            let jolt_verifier::CommitmentPayload::Dory(commitments) = &mut case.proof.commitments
            else {
                panic!("core fixture should carry Dory commitments");
            };
            let _ = commitments.ra.bytecode.pop();
        },
    );
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.extra"),
        &base,
        |case| {
            let jolt_verifier::CommitmentPayload::Dory(commitments) = &mut case.proof.commitments
            else {
                panic!("core fixture should carry Dory commitments");
            };
            commitments.ra.bytecode.push(commitments.ram_inc.clone());
        },
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tampered_joint_opening_proof_reject() {
    let base = core_fixtures::standard_muldiv_case();
    let replacement = core_fixtures::standard_fibonacci_small_case()
        .proof
        .joint_opening_proof;
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.joint_opening_proof"),
        &base,
        |case| {
            case.proof.joint_opening_proof = replacement;
        },
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "direct commitment tampering fixtures are not wired yet"]
fn tampered_commitment_order_reject() {}
