#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

mod support;

use jolt_blindfold::VerificationError;
use jolt_poly::CompressedPoly;
use jolt_transcript::{prover_transcript, Blake2b512};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

fn verify_blindfold_protocol_pipeline(
    full: &BlindFoldTestProof,
) -> Result<(), VerificationError<F>> {
    let mut transcript = prover_transcript(
        b"protocol-backed-blindfold-proof",
        [0u8; 32],
        Blake2b512::default(),
    );
    append_protocol_transcript_prefix(&full.protocol, &mut transcript);
    full.protocol
        .verify::<VC, _>(&full.proof, &full.setup, &mut transcript)
}

#[test]
fn blindfold_protocol_pipeline_verifies_committed_sumcheck_outputs_and_eval_commitments() {
    let mut rng = ChaCha20Rng::from_seed([81; 32]);
    let full = prove_blindfold_protocol_pipeline(&mut rng);

    assert!(full.protocol.dimensions.coefficient_rows > 0);
    assert!(full.protocol.dimensions.output_claim_rows > 0);
    assert!(full.protocol.dimensions.auxiliary_rows > 0);
    assert!(!full.protocol.eval_commitments.is_empty());
    verify_blindfold_protocol_pipeline(&full).expect("protocol-backed BlindFold proof verifies");
}

#[test]
fn blindfold_protocol_pipeline_randomness_is_empirically_independent() {
    const SAMPLES: usize = 128;
    let mut rng = ChaCha20Rng::from_seed([61; 32]);
    let mut projections = [
        StatisticalProjection::new("random_u", SAMPLES),
        StatisticalProjection::new("auxiliary_commitment", SAMPLES),
        StatisticalProjection::new("random_round_commitment", SAMPLES),
        StatisticalProjection::new("random_error_commitment", SAMPLES),
        StatisticalProjection::new("cross_term_commitment", SAMPLES),
        StatisticalProjection::new("outer_sumcheck", SAMPLES),
        StatisticalProjection::new("inner_sumcheck", SAMPLES),
        StatisticalProjection::new("witness_opening", SAMPLES),
        StatisticalProjection::new("error_opening", SAMPLES),
    ];

    for _ in 0..SAMPLES {
        let full = prove_blindfold_protocol_pipeline(&mut rng);

        verify_blindfold_protocol_pipeline(&full).expect("sample proof verifies");

        let values = [
            field_low_u64(full.proof.random_u),
            transcript_projection(&full.proof.auxiliary_row_commitments[0]),
            transcript_projection(&full.proof.random_round_commitments[0]),
            transcript_projection(&full.proof.random_error_row_commitments[0]),
            transcript_projection(&full.proof.cross_term_error_row_commitments[0]),
            compressed_sumcheck_projection(&full.proof.outer_sumcheck),
            compressed_sumcheck_projection(&full.proof.inner_sumcheck),
            opening_projection(&full.proof.witness_opening),
            opening_projection(&full.proof.error_opening),
        ];

        for (projection, value) in projections.iter_mut().zip(values) {
            projection.push(value);
        }
    }

    for projection in &projections {
        assert_empirical_distribution(projection);
    }
    assert_empirical_pairwise_independence(&projections[0], &projections[1]);
    assert_empirical_pairwise_independence(&projections[0], &projections[7]);
    assert_empirical_pairwise_independence(&projections[1], &projections[2]);
    assert_empirical_pairwise_independence(&projections[3], &projections[4]);
    assert_empirical_pairwise_independence(&projections[5], &projections[6]);
    assert_empirical_pairwise_independence(&projections[7], &projections[8]);
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_random_u() {
    let mut rng = ChaCha20Rng::from_seed([71; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.random_u += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_outer_sumcheck() {
    let mut rng = ChaCha20Rng::from_seed([72; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    let mut coefficients = full.proof.outer_sumcheck.round_polynomials[0]
        .coeffs_except_linear_term()
        .to_vec();
    coefficients[0] += f(1);
    full.proof.outer_sumcheck.round_polynomials[0] = CompressedPoly::new(coefficients);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_matrix_eval() {
    let mut rng = ChaCha20Rng::from_seed([73; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.az_rx += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_witness_opening() {
    let mut rng = ChaCha20Rng::from_seed([74; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.witness_opening.combined_vector[0] += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_error_opening_blinding() {
    let mut rng = ChaCha20Rng::from_seed([75; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.error_opening.combined_blinding += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_wrong_transcript() {
    let mut rng = ChaCha20Rng::from_seed([76; 32]);
    let full = prove_blindfold_protocol_pipeline(&mut rng);
    let mut transcript = prover_transcript(b"wrong-transcript", [0u8; 32], Blake2b512::default());
    append_protocol_transcript_prefix(&full.protocol, &mut transcript);

    assert!(full
        .protocol
        .verify::<VC, _>(&full.proof, &full.setup, &mut transcript)
        .is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_random_commitment_row() {
    let mut rng = ChaCha20Rng::from_seed([77; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.random_round_commitments.swap(0, 1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_inner_sumcheck() {
    let mut rng = ChaCha20Rng::from_seed([78; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    let mut coefficients = full.proof.inner_sumcheck.round_polynomials[1]
        .coeffs_except_linear_term()
        .to_vec();
    coefficients[0] += f(1);
    full.proof.inner_sumcheck.round_polynomials[1] = CompressedPoly::new(coefficients);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_truncated_error_rows_before_opening_checks() {
    let mut rng = ChaCha20Rng::from_seed([79; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    let _ = full.proof.random_error_row_commitments.pop();

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_opening() {
    let mut rng = ChaCha20Rng::from_seed([82; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.folded_eval_outputs[0] += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_blinding() {
    let mut rng = ChaCha20Rng::from_seed([87; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.folded_eval_blindings[0] += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_witness_opening() {
    let mut rng = ChaCha20Rng::from_seed([85; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.folded_eval_output_openings[0].combined_vector[0] += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_blinding_witness_opening() {
    let mut rng = ChaCha20Rng::from_seed([86; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.folded_eval_blinding_openings[0].combined_vector[0] += f(1);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_final_eval_openings_use_dedicated_rows() {
    let mut rng = ChaCha20Rng::from_seed([88; 32]);
    let full = prove_blindfold_protocol_pipeline(&mut rng);
    let coordinates = full
        .protocol
        .final_opening_witness_coordinates()
        .expect("final opening coordinates are valid");
    let eval = coordinates[0]
        .evaluation
        .expect("final opening has an evaluation coordinate");
    let blinding = coordinates[0]
        .blinding
        .expect("final opening has a blinding coordinate");

    assert_eq!(eval.column, 0);
    assert_eq!(blinding.column, 0);
    assert_ne!(eval.row, blinding.row);
    assert_eq!(
        full.proof.folded_eval_output_openings[0].combined_vector[0],
        full.proof.folded_eval_outputs[0]
    );
    assert_eq!(
        full.proof.folded_eval_blinding_openings[0].combined_vector[0],
        full.proof.folded_eval_blindings[0]
    );
    assert!(
        full.proof.folded_eval_output_openings[0].combined_vector[1..]
            .iter()
            .all(|value| *value == f(0))
    );
    assert!(
        full.proof.folded_eval_blinding_openings[0].combined_vector[1..]
            .iter()
            .all(|value| *value == f(0))
    );
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_auxiliary_commitment() {
    let mut rng = ChaCha20Rng::from_seed([83; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.proof.auxiliary_row_commitments[0] = full.proof.random_round_commitments[0];

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_output_claim_row_commitment() {
    let mut rng = ChaCha20Rng::from_seed([84; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.protocol.committed_output_claims[0].commitments[0] =
        full.proof.random_round_commitments[0];

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}
