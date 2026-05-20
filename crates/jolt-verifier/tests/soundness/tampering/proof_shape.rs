#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use crate::support;
#[cfg(feature = "core-fixtures")]
use crate::support::tamper_manifest;
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_verifier::compat::claims::attach_empty_opening_claims;
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_verifier::proof::JoltProofClaims;

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn mixed_clear_and_committed_stage_proofs_reject_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.stages.clear_vs_committed"),
        &base,
        |case| {
            case.proof.stages.stage5_sumcheck_proof = jolt_sumcheck::SumcheckProof::Committed(
                jolt_sumcheck::CommittedSumcheckProof::default(),
            );
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn mixed_uniskip_stage_proof_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.stages.clear_vs_committed"),
        &base,
        |case| {
            case.proof.stages.stage1_uni_skip_first_round_proof =
                jolt_sumcheck::SumcheckProof::Committed(
                    jolt_sumcheck::CommittedSumcheckProof::default(),
                );
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn zk_claim_payload_in_clear_mode_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.claims.mode_payload"),
        &base,
        |case| {
            case.proof.claims = JoltProofClaims::Zk {
                blindfold_proof: (),
            };
        },
    );
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
#[ignore = "real ZK core fixture tampering is deferred until the ZK verifier frontier"]
fn unexpected_zk_opening_claims_reject_now() {
    assert_zk_target_active("proof.claims.mode_payload");
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    attach_empty_opening_claims(&mut case.proof);

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
#[test]
#[ignore = "real ZK core fixture tampering is deferred until the ZK verifier frontier"]
fn clear_stage_in_zk_proof_rejects_now() {
    assert_zk_target_active("proof.stages.clear_vs_committed");
    let mut case = crate::support::core_fixtures::zk_muldiv_case();
    case.proof.stages.stage3_sumcheck_proof = jolt_sumcheck::SumcheckProof::Clear(
        jolt_sumcheck::ClearProof::Compressed(jolt_sumcheck::CompressedSumcheckProof::default()),
    );

    support::assert_zk_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures to live-generate, cast, and tamper real core proofs"]
fn tampered_mixed_proof_shape_reject() {
    assert_eq!(
        soundness_expectation(tampering::MIXED_PROOF_SHAPE),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_core_case() -> crate::support::core_fixtures::CoreVerifierCase {
    crate::support::core_fixtures::standard_muldiv_case()
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn assert_zk_target_active(name: &str) {
    let target = tamper_manifest::required_target(name);
    tamper_manifest::assert_manifest_target_is_active(target);
    assert!(
        target.mode.includes(true),
        "tamper target mode does not include ZK: {target:?}"
    );
}
