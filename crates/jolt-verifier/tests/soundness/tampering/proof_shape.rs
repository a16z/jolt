#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support::proof_claims::attach_empty_opening_claims;
#[cfg(feature = "prover-fixtures")]
use crate::support::tamper_manifest;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_poly::CompressedPoly;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_sumcheck::{ClearProof, CompressedSumcheckProof, SumcheckProof};
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use jolt_verifier::proof::JoltProofClaims;
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use {
    jolt_algebra::{Fr, Ring},
    jolt_crypto::{Bn254G1, VectorCommitmentOpening},
    jolt_sumcheck::CompressedSumcheckProof,
    jolt_verifier::BlindFoldZkProof,
    jolt_zk::blindfold::{BlindFoldProof, FoldedEvaluation, FoldingCommitments, OuterClaims},
    jolt_zk::masking::MaskedMessages,
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
fn zk_claim_payload_in_clear_mode_rejects_now() {
    let base = crate::support::verifier_fixtures::standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.claims.mode_payload"),
        &base,
        |case| {
            case.proof.claims = JoltProofClaims::Zk {
                masked: MaskedMessages::default(),
                zk_proof: BlindFoldZkProof {
                    pad_commitment: Vec::new(),
                    blindfold: empty_blindfold_proof(),
                },
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
    // A masked proof carries its rounds on the masked tape; clear rounds in a
    // stage proof are the other mode's shape.
    case.proof.stages.stage3_sumcheck_proof =
        SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials: vec![CompressedPoly::new(vec![
                <jolt_algebra::Fr as jolt_algebra::Ring>::from_u64(1),
            ])],
            pow_nonces: Vec::new(),
        }));

    support::assert_zk_rejects(case.verify());
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn empty_blindfold_proof() -> BlindFoldProof<Fr, Bn254G1> {
    let zero = Fr::from_u64(0);
    let opening = || VectorCommitmentOpening {
        combined_vector: Vec::new(),
        combined_blinding: zero,
    };
    BlindFoldProof {
        folding: FoldingCommitments {
            auxiliary_rows: Vec::new(),
            random_u: zero,
            random_pad_rows: Vec::new(),
            random_auxiliary_rows: Vec::new(),
            random_error_rows: Vec::new(),
            random_eval: Bn254G1::default(),
            cross_term_error_rows: Vec::new(),
        },
        folded_evaluation: FoldedEvaluation {
            output: zero,
            blinding: zero,
            output_opening: opening(),
            blinding_opening: opening(),
        },
        outer_sumcheck: CompressedSumcheckProof::default(),
        outer_claims: OuterClaims {
            abc: [zero; 3],
            error_opening: opening(),
        },
        inner_sumcheck: CompressedSumcheckProof::default(),
        witness_opening: opening(),
    }
}

#[cfg(feature = "prover-fixtures")]
#[test]
fn committed_program_rejects_mismatched_trace_order() {
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_verifier::VerifierError;

    #[cfg(not(feature = "zk"))]
    let mut case = crate::support::verifier_fixtures::standard_committed_muldiv_case();
    #[cfg(feature = "zk")]
    let mut case = crate::support::verifier_fixtures::zk_committed_muldiv_case();
    case.proof.trace_polynomial_order = TracePolynomialOrder::AddressMajor;
    assert!(matches!(
        case.verify(),
        Err(VerifierError::InvalidCommittedProgram { .. })
    ));
}
