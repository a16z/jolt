#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

mod support;

use jolt_blindfold::VerificationError;
use jolt_transcript::{prover_transcript, verifier_transcript, Blake2b512, FsAbsorb, FsChallenge};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

fn verify_blindfold_protocol_pipeline(
    full: &BlindFoldTestProof,
) -> Result<(), VerificationError<F>> {
    let mut transcript = verifier_transcript(
        b"protocol-backed-blindfold-proof",
        [0u8; 32],
        Blake2b512::default(),
        &full.narg,
    );
    append_protocol_transcript_prefix(&full.protocol, &mut transcript);
    full.protocol
        .verify_from_narg::<VC, _>(&full.setup, &mut transcript)?;
    transcript
        .check_eof()
        .map_err(|_| VerificationError::MalformedNarg {
            name: "trailing BlindFold NARG",
        })
}

fn tamper_narg_at(narg: &mut [u8], numerator: usize, denominator: usize) {
    assert!(!narg.is_empty(), "BlindFold proof must carry NARG bytes");
    let index = (narg.len() * numerator / denominator).min(narg.len() - 1);
    narg[index] ^= 1;
}

fn narg_projection(narg: &[u8], domain: u8) -> u64 {
    let mut transcript = prover_transcript(
        b"blindfold-narg-statistical-projection",
        [domain; 32],
        Blake2b512::default(),
    );
    transcript.absorb_bytes(narg);
    field_low_u64(FsChallenge::<F>::challenge(&mut transcript))
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
        StatisticalProjection::new("narg_projection_0", SAMPLES),
        StatisticalProjection::new("narg_projection_1", SAMPLES),
        StatisticalProjection::new("narg_projection_2", SAMPLES),
        StatisticalProjection::new("narg_projection_3", SAMPLES),
        StatisticalProjection::new("narg_projection_4", SAMPLES),
        StatisticalProjection::new("narg_projection_5", SAMPLES),
        StatisticalProjection::new("narg_projection_6", SAMPLES),
        StatisticalProjection::new("narg_projection_7", SAMPLES),
        StatisticalProjection::new("narg_projection_8", SAMPLES),
    ];

    for _ in 0..SAMPLES {
        let full = prove_blindfold_protocol_pipeline(&mut rng);

        verify_blindfold_protocol_pipeline(&full).expect("sample proof verifies");

        let values =
            core::array::from_fn::<_, 9, _>(|index| narg_projection(&full.narg, index as u8));

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
    tamper_narg_at(&mut full.narg, 1, 8);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_outer_sumcheck() {
    let mut rng = ChaCha20Rng::from_seed([72; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 5, 8);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_matrix_eval() {
    let mut rng = ChaCha20Rng::from_seed([73; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 6, 8);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_witness_opening() {
    let mut rng = ChaCha20Rng::from_seed([74; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 7, 8);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_error_opening_blinding() {
    let mut rng = ChaCha20Rng::from_seed([75; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 13, 16);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_wrong_transcript() {
    let mut rng = ChaCha20Rng::from_seed([76; 32]);
    let full = prove_blindfold_protocol_pipeline(&mut rng);
    let mut transcript = verifier_transcript(
        b"wrong-transcript",
        [0u8; 32],
        Blake2b512::default(),
        &full.narg,
    );
    append_protocol_transcript_prefix(&full.protocol, &mut transcript);

    assert!(full
        .protocol
        .verify_from_narg::<VC, _>(&full.setup, &mut transcript)
        .is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_random_commitment_row() {
    let mut rng = ChaCha20Rng::from_seed([77; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 3, 16);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_inner_sumcheck() {
    let mut rng = ChaCha20Rng::from_seed([78; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 11, 16);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_truncated_error_rows_before_opening_checks() {
    let mut rng = ChaCha20Rng::from_seed([79; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.narg.truncate(full.narg.len() / 2);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_opening() {
    let mut rng = ChaCha20Rng::from_seed([82; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 7, 16);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_blinding() {
    let mut rng = ChaCha20Rng::from_seed([87; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 1, 2);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_witness_opening() {
    let mut rng = ChaCha20Rng::from_seed([85; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 9, 16);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_folded_eval_blinding_witness_opening() {
    let mut rng = ChaCha20Rng::from_seed([86; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 10, 16);

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
    verify_blindfold_protocol_pipeline(&full).expect("dedicated final opening rows verify");
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_auxiliary_commitment() {
    let mut rng = ChaCha20Rng::from_seed([83; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    tamper_narg_at(&mut full.narg, 1, 16);

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}

#[test]
fn blindfold_protocol_pipeline_rejects_tampered_output_claim_row_commitment() {
    let mut rng = ChaCha20Rng::from_seed([84; 32]);
    let mut full = prove_blindfold_protocol_pipeline(&mut rng);
    full.protocol.committed_output_claims[0].commitments[0] =
        full.protocol.sumcheck_consistency[0].rounds[0].commitment;

    assert!(verify_blindfold_protocol_pipeline(&full).is_err());
}
