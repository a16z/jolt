use crate::{
    soundness::tampering,
    support::dory_pedersen,
    support::tamper_manifest,
    support::{soundness_expectation, HarnessExpectation},
};
use jolt_verifier::{compat::claims::attach_empty_opening_claims, proof::JoltProofClaims};

#[test]
fn mixed_clear_and_committed_stage_proofs_reject_now() {
    tamper_manifest::assert_dory_tamper_rejects(
        tamper_manifest::required_target("proof.stages.clear_vs_committed"),
        dory_pedersen::standard_case(),
        |case| {
            case.proof.stages.stage5_sumcheck_proof = jolt_sumcheck::SumcheckProof::Committed(
                jolt_sumcheck::CommittedSumcheckProof::default(),
            );
        },
    );
}

#[test]
fn mixed_uniskip_stage_proof_rejects_now() {
    tamper_manifest::assert_dory_tamper_rejects(
        tamper_manifest::required_target("proof.stages.clear_vs_committed"),
        dory_pedersen::standard_case(),
        |case| {
            case.proof.stages.stage1_uni_skip_first_round_proof =
                jolt_sumcheck::SumcheckProof::Committed(
                    jolt_sumcheck::CommittedSumcheckProof::default(),
                );
        },
    );
}

#[test]
fn zk_claim_payload_in_transparent_mode_rejects_now() {
    tamper_manifest::assert_dory_tamper_rejects(
        tamper_manifest::required_target("proof.claims.mode_payload"),
        dory_pedersen::standard_case(),
        |case| {
            case.proof.claims = JoltProofClaims::Zk {
                blindfold_proof: (),
            };
        },
    );
}

#[test]
fn unexpected_standard_blindfold_proof_rejects_now() {
    tamper_manifest::assert_dory_tamper_rejects(
        tamper_manifest::required_target("proof.claims.mode_payload"),
        dory_pedersen::standard_case(),
        |case| {
            case.proof.claims = JoltProofClaims::Zk {
                blindfold_proof: (),
            };
        },
    );
}

#[test]
fn unexpected_zk_opening_claims_reject_now() {
    tamper_manifest::assert_dory_tamper_rejects(
        tamper_manifest::required_target("proof.claims.mode_payload"),
        dory_pedersen::zk_case(),
        |case| {
            attach_empty_opening_claims(&mut case.proof);
        },
    );
}

#[test]
fn clear_stage_in_zk_proof_rejects_now() {
    tamper_manifest::assert_dory_tamper_rejects(
        tamper_manifest::required_target("proof.stages.clear_vs_committed"),
        dory_pedersen::zk_case(),
        |case| {
            case.proof.stages.stage3_sumcheck_proof =
                jolt_sumcheck::SumcheckProof::Clear(jolt_sumcheck::ClearProof::Compressed(
                    jolt_sumcheck::CompressedSumcheckProof::default(),
                ));
        },
    );
}

#[test]
#[ignore = "direct proof-shape tampering fixtures are not wired yet"]
fn tampered_mixed_proof_shape_reject() {
    assert_eq!(
        soundness_expectation(tampering::MIXED_PROOF_SHAPE),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}
