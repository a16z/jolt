#![cfg_attr(
    all(feature = "core-fixtures", feature = "zk"),
    expect(
        clippy::expect_used,
        clippy::panic,
        reason = "fixture audit tests should fail loudly when core proof shape assumptions break"
    )
)]

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use crate::support;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_dory::DoryScheme;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_field::Fr;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_sumcheck::SumcheckProof;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_transcript::Blake2bTranscript;
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
use jolt_verifier::JoltProofClaims;

#[test]
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn zk_muldiv_core_proof_is_accepted() {
    support::assert_zk_accepts(crate::support::core_fixtures::zk_muldiv_case().verify());
}

#[test]
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn zk_muldiv_blindfold_shape_audit_matches_modular_protocol() {
    let case = crate::support::core_fixtures::zk_muldiv_case();
    let modular = jolt_verifier::audit_zk_blindfold_protocol_shape::<
        Fr,
        DoryScheme,
        Pedersen<Bn254G1>,
        Blake2bTranscript,
        _,
    >(&case.preprocessing, &case.public_io, &case.proof, None)
    .expect("build modular BlindFold protocol shape");
    let JoltProofClaims::Zk { blindfold_proof } = &case.proof.claims else {
        panic!("ZK core fixture must carry a BlindFold proof");
    };
    let legacy = blindfold_proof.shape();
    let committed_round_rows =
        committed_round_rows(&case.proof.stages.stage1_uni_skip_first_round_proof)
            + committed_round_rows(&case.proof.stages.stage1_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage2_uni_skip_first_round_proof)
            + committed_round_rows(&case.proof.stages.stage2_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage3_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage4_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage5_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage6_sumcheck_proof)
            + committed_round_rows(&case.proof.stages.stage7_sumcheck_proof);
    let committed_output_claim_rows =
        committed_output_claim_rows(&case.proof.stages.stage1_uni_skip_first_round_proof)
            + committed_output_claim_rows(&case.proof.stages.stage1_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage2_uni_skip_first_round_proof)
            + committed_output_claim_rows(&case.proof.stages.stage2_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage3_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage4_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage5_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage6_sumcheck_proof)
            + committed_output_claim_rows(&case.proof.stages.stage7_sumcheck_proof);

    assert_eq!(committed_round_rows, modular.coefficient_rows);
    assert_eq!(
        legacy.random_round_commitment_rows,
        modular.coefficient_rows
    );
    assert_eq!(committed_output_claim_rows, modular.output_claim_rows);
    assert_eq!(legacy.random_output_claim_rows, modular.output_claim_rows);
    assert_eq!(legacy.random_eval_commitments, modular.eval_commitments);
    assert_eq!(legacy.auxiliary_rows, modular.auxiliary_rows);
    assert_eq!(legacy.random_auxiliary_rows, modular.auxiliary_rows);
    assert_eq!(legacy.random_error_rows, modular.error_row_count);
    assert_eq!(legacy.cross_term_error_rows, modular.error_row_count);
    assert_eq!(legacy.folded_eval_output_openings, modular.eval_commitments);
    assert_eq!(
        legacy.folded_eval_blinding_openings,
        modular.eval_commitments
    );

    assert_eq!(modular.coefficient_rows, 222);
    assert_eq!(modular.output_claim_rows, 14);
    assert_eq!(modular.eval_commitments, 1);
    assert_eq!(modular.auxiliary_rows, 33);
    assert_eq!(modular.error_row_count, 64);
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn committed_round_rows<F, C>(proof: &SumcheckProof<F, C>) -> usize
where
    F: jolt_field::Field,
{
    proof
        .as_committed()
        .expect("ZK core fixture must use committed sumcheck proofs")
        .rounds
        .len()
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn committed_output_claim_rows<F, C>(proof: &SumcheckProof<F, C>) -> usize
where
    F: jolt_field::Field,
{
    proof
        .as_committed()
        .expect("ZK core fixture must use committed sumcheck proofs")
        .output_claims
        .commitments
        .len()
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), not(feature = "zk")))]
#[ignore = "enable --features core-fixtures,zk to live-generate and cast this core ZK fixture"]
fn zk_muldiv_core_proof_is_accepted() {}

#[test]
#[ignore = "prefix BlindFold fixture generation is not wired yet"]
fn zk_stage1_prefix_is_accepted() {}
