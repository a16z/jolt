#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::expect_used,
        clippy::panic,
        reason = "fixture audit tests should fail loudly when verifier object shape assumptions break"
    )
)]

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_dory::DoryScheme;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_field::Fr;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_sumcheck::SumcheckProof;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_transcript::Blake2b512;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_verifier::JoltProofClaims;

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_muldiv_verifier_proof_is_accepted() {
    support::assert_zk_accepts(crate::support::verifier_fixtures::zk_muldiv_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_committed_muldiv_verifier_proof_is_accepted() {
    support::assert_zk_accepts(
        crate::support::verifier_fixtures::zk_committed_muldiv_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_committed_muldiv_blindfold_shape_audit_matches_modular_protocol() {
    let case = crate::support::verifier_fixtures::zk_committed_muldiv_case();
    let modular = support::zk_audit::audit_zk_blindfold_protocol_shape::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2b512,
        _,
    >(&case.preprocessing, &case.public_io, &case.proof, None)
    .expect("build modular BlindFold protocol shape");
    let JoltProofClaims::Zk { blindfold_proof } = &case.proof.claims else {
        panic!("ZK verifier fixture must carry a BlindFold proof");
    };
    let proof_shape = blindfold_proof_shape(blindfold_proof);

    assert_eq!(
        proof_shape.random_round_commitment_rows,
        modular.coefficient_rows
    );
    assert_eq!(
        proof_shape.random_output_claim_rows,
        modular.output_claim_rows
    );
    assert_eq!(
        proof_shape.random_eval_commitments,
        modular.eval_commitments
    );
    assert_eq!(proof_shape.auxiliary_rows, modular.auxiliary_rows);
    assert_eq!(proof_shape.random_auxiliary_rows, modular.auxiliary_rows);
    assert_eq!(proof_shape.random_error_rows, modular.error_row_count);
    assert_eq!(proof_shape.cross_term_error_rows, modular.error_row_count);
}

#[test]
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn zk_muldiv_blindfold_shape_audit_matches_modular_protocol() {
    let case = crate::support::verifier_fixtures::zk_muldiv_case();
    let modular = support::zk_audit::audit_zk_blindfold_protocol_shape::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2b512,
        _,
    >(&case.preprocessing, &case.public_io, &case.proof, None)
    .expect("build modular BlindFold protocol shape");
    let JoltProofClaims::Zk { blindfold_proof } = &case.proof.claims else {
        panic!("ZK verifier fixture must carry a BlindFold proof");
    };
    let proof_shape = blindfold_proof_shape(blindfold_proof);
    let committed_round_rows =
        committed_round_rows(&case.proof.stages.stage1_uni_skip_first_round_proof)
            + committed_round_rows(&case.proof.stages.stage1_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage2_uni_skip_first_round_proof)
            + committed_round_rows(&case.proof.stages.stage2_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage3_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage4_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage5_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage6a_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage6b_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage7_sumcheck_proof);
    let committed_output_claim_rows =
        committed_output_claim_rows(&case.proof.stages.stage1_uni_skip_first_round_proof)
            + committed_output_claim_rows(&case.proof.stages.stage1_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage2_uni_skip_first_round_proof)
            + committed_output_claim_rows(&case.proof.stages.stage2_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage3_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage4_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage5_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage6a_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage6b_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage7_sumcheck_proof);

    assert_eq!(committed_round_rows, modular.coefficient_rows);
    assert_eq!(
        proof_shape.random_round_commitment_rows,
        modular.coefficient_rows
    );
    assert_eq!(committed_output_claim_rows, modular.output_claim_rows);
    assert_eq!(
        proof_shape.random_output_claim_rows,
        modular.output_claim_rows
    );
    assert_eq!(
        proof_shape.random_eval_commitments,
        modular.eval_commitments
    );
    assert_eq!(proof_shape.auxiliary_rows, modular.auxiliary_rows);
    assert_eq!(proof_shape.random_auxiliary_rows, modular.auxiliary_rows);
    assert_eq!(proof_shape.random_error_rows, modular.error_row_count);
    assert_eq!(proof_shape.cross_term_error_rows, modular.error_row_count);
    assert_eq!(
        proof_shape.folded_eval_output_openings,
        modular.eval_commitments
    );
    assert_eq!(
        proof_shape.folded_eval_blinding_openings,
        modular.eval_commitments
    );

    assert_eq!(modular.coefficient_rows, 221);
    assert_eq!(modular.output_claim_rows, 15);
    assert_eq!(modular.eval_commitments, 1);
    assert_eq!(modular.auxiliary_rows, 33);
    assert_eq!(modular.error_row_count, 64);
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlindFoldProofShape {
    auxiliary_rows: usize,
    random_round_commitment_rows: usize,
    random_output_claim_rows: usize,
    random_auxiliary_rows: usize,
    random_error_rows: usize,
    random_eval_commitments: usize,
    cross_term_error_rows: usize,
    folded_eval_output_openings: usize,
    folded_eval_blinding_openings: usize,
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn blindfold_proof_shape(
    proof: &jolt_blindfold::BlindFoldProof<Fr, Bn254G1>,
) -> BlindFoldProofShape {
    BlindFoldProofShape {
        auxiliary_rows: proof.auxiliary_row_commitments.len(),
        random_round_commitment_rows: proof.random_round_commitments.len(),
        random_output_claim_rows: proof.random_output_claim_row_commitments.len(),
        random_auxiliary_rows: proof.random_auxiliary_row_commitments.len(),
        random_error_rows: proof.random_error_row_commitments.len(),
        random_eval_commitments: proof.random_eval_commitments.len(),
        cross_term_error_rows: proof.cross_term_error_row_commitments.len(),
        folded_eval_output_openings: proof.folded_eval_output_openings.len(),
        folded_eval_blinding_openings: proof.folded_eval_blinding_openings.len(),
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn committed_round_rows<F, C>(proof: &SumcheckProof<F, C>) -> usize
where
    F: jolt_field::Field,
{
    proof
        .as_committed()
        .expect("ZK verifier fixture must use committed sumcheck proofs")
        .rounds
        .len()
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn committed_output_claim_rows<F, C>(proof: &SumcheckProof<F, C>) -> usize
where
    F: jolt_field::Field,
{
    proof
        .as_committed()
        .expect("ZK verifier fixture must use committed sumcheck proofs")
        .output_claims
        .commitments
        .len()
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), not(feature = "zk")))]
#[ignore = "enable --features prover-fixtures,zk to live-generate this verifier ZK fixture"]
fn zk_muldiv_verifier_proof_is_accepted() {}

#[test]
#[ignore = "prefix BlindFold fixture generation is not wired yet"]
fn zk_stage1_prefix_is_accepted() {}
