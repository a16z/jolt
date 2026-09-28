#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::{tamper_manifest, verifier_fixtures};

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn tampered_commitment_order_reject() {
    let base = verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.order"),
        &base,
        |case| {
            std::mem::swap(
                &mut case.proof.commitments.rd_inc,
                &mut case.proof.commitments.ram_inc,
            );
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
            case.proof.commitments.rd_inc = case.proof.commitments.ram_inc.clone();
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
            let _ = case.proof.commitments.bytecode_ra.pop();
        },
    );
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.commitments.extra"),
        &base,
        |case| {
            case.proof
                .commitments
                .bytecode_ra
                .push(case.proof.commitments.ram_inc.clone());
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
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn replaced_untrusted_advice_commitment_rejects() {
    let base = verifier_fixtures::standard_advice_consumer_case();
    assert!(
        base.proof.untrusted_advice_commitment.is_some(),
        "advice-consumer fixture must carry an untrusted advice commitment"
    );
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.untrusted_advice_commitment"),
        &base,
        |case| {
            // valid-but-wrong payload: another commitment from the same proof
            case.proof.untrusted_advice_commitment = Some(case.proof.commitments.rd_inc.clone());
        },
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn replaced_trusted_advice_commitment_rejects() {
    let base = verifier_fixtures::standard_advice_consumer_case();
    assert!(
        base.trusted_advice_commitment.is_some(),
        "advice-consumer fixture must carry a trusted advice commitment"
    );
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("trusted_advice_commitment"),
        &base,
        |case| {
            case.trusted_advice_commitment = Some(case.proof.commitments.rd_inc.clone());
        },
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "direct commitment tampering fixtures are not wired yet"]
fn tampered_commitment_order_reject() {}
