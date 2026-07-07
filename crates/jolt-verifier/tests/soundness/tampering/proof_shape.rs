#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support::proof_claims::attach_empty_opening_claims;
#[cfg(feature = "prover-fixtures")]
use crate::support::tamper_manifest;
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_verifier::proof::JoltProofClaims;
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use {
    jolt_blindfold::BlindFoldProof,
    jolt_crypto::VectorCommitmentOpening,
    jolt_field::{Fr, FromPrimitiveInt},
    jolt_sumcheck::CompressedSumcheckProof,
};

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn mixed_clear_and_committed_stage_proofs_reject_now() {
    let base = crate::support::verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.stages.clear_vs_committed"),
        &base,
        |case| {
            case.proof.stages.stage5_sumcheck_proof = jolt_sumcheck::SumcheckProof::Committed(
                jolt_sumcheck::CommittedSumcheckProof::default(),
            );
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn mixed_uniskip_stage_proof_rejects_now() {
    let base = crate::support::verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn zk_claim_payload_in_clear_mode_rejects_now() {
    let base = crate::support::verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.claims.mode_payload"),
        &base,
        |case| {
            case.proof.claims = JoltProofClaims::Zk {
                blindfold_proof: empty_blindfold_proof(),
            };
        },
    );
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn unexpected_zk_opening_claims_reject_now() {
    tamper_manifest::assert_zk_target_active("proof.claims.mode_payload");
    let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
    attach_empty_opening_claims(&mut case.proof);

    support::assert_zk_rejects(case.verify());
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn clear_stage_in_zk_proof_rejects_now() {
    tamper_manifest::assert_zk_target_active("proof.stages.clear_vs_committed");
    let mut case = crate::support::verifier_fixtures::zk_muldiv_case();
    case.proof.stages.stage3_sumcheck_proof = jolt_sumcheck::SumcheckProof::Clear(
        jolt_sumcheck::ClearProof::Compressed(jolt_sumcheck::CompressedSumcheckProof::default()),
    );

    support::assert_zk_rejects(case.verify());
}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures to live-generate and tamper verifier-native proofs"]
fn tampered_mixed_proof_shape_reject() {}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn empty_blindfold_proof() -> BlindFoldProof<Fr, jolt_crypto::Bn254G1> {
    let zero = Fr::from_u64(0);
    let opening = VectorCommitmentOpening {
        combined_vector: Vec::new(),
        combined_blinding: zero,
    };
    BlindFoldProof {
        auxiliary_row_commitments: Vec::new(),
        random_round_commitments: Vec::new(),
        random_output_claim_row_commitments: Vec::new(),
        random_auxiliary_row_commitments: Vec::new(),
        random_error_row_commitments: Vec::new(),
        random_eval_commitments: Vec::new(),
        random_u: zero,
        cross_term_error_row_commitments: Vec::new(),
        outer_sumcheck: CompressedSumcheckProof::default(),
        az_rx: zero,
        bz_rx: zero,
        cz_rx: zero,
        inner_sumcheck: CompressedSumcheckProof::default(),
        witness_opening: opening.clone(),
        error_opening: opening.clone(),
        folded_eval_outputs: Vec::new(),
        folded_eval_blindings: Vec::new(),
        folded_eval_output_openings: Vec::new(),
        folded_eval_blinding_openings: Vec::new(),
    }
}
