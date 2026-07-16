//! Integration tests for the Dory commitment scheme.
//!
//! Public-API-only tests — no `pub(crate)` imports. Exercises commit, open,
//! verify, streaming, combine, and negative cases across transcript backends.

#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

use dory::backends::arkworks::ArkG1;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, StreamingCommitment, ZkOpeningScheme,
};
use jolt_poly::{OneHotPolynomial, Polynomial};
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn round_trip<T: Transcript<Challenge = Fr>>(num_vars: usize, seed: u64, label: &'static [u8]) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

    // With hint
    let mut pt = T::new(label);
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt).unwrap();

    let mut vt = T::new(label);
    DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt)
        .expect("round-trip verification (with hint) must succeed");

    // Without hint
    let mut pt2 = T::new(label);
    let proof2 = DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut pt2).unwrap();

    let mut vt2 = T::new(label);
    DoryScheme::verify(
        &commitment,
        &point,
        eval,
        &proof2,
        &verifier_setup,
        &mut vt2,
    )
    .expect("round-trip verification (without hint) must succeed");
}

#[test]
fn commit_open_verify_various_sizes() {
    for num_vars in [2, 3, 4, 6] {
        round_trip::<Blake2bTranscript>(num_vars, 100 + num_vars as u64, b"cov-sizes");
    }
}

#[test]
fn commit_open_verify_both_transcripts() {
    let num_vars = 4;
    round_trip::<Blake2bTranscript>(num_vars, 200, b"blake2b-rt");
    round_trip::<KeccakTranscript>(num_vars, 200, b"keccak-rt");
}

#[test]
fn streaming_equals_direct_various_sizes() {
    for num_vars in [2usize, 4, 6] {
        let sigma = num_vars.div_ceil(2);
        let num_cols = 1usize << sigma;
        let mut rng = ChaCha20Rng::seed_from_u64(300 + num_vars as u64);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

        let (direct, _) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

        let mut partial = DoryScheme::begin(&prover_setup);
        for row in poly.evaluations().chunks(num_cols) {
            DoryScheme::feed(&mut partial, row, &prover_setup);
        }
        let streamed = DoryScheme::finish(partial, &prover_setup);

        assert_eq!(
            direct, streamed,
            "streaming and direct must match for num_vars={num_vars}"
        );
    }
}

#[test]
fn one_hot_commitment_matches_dense() {
    let num_vars = 4;
    let k = 4;
    let indices = vec![Some(2), None, Some(0), Some(3)];
    let mut evals = vec![Fr::from_u64(0); 1 << num_vars];
    for (row, col) in indices.iter().enumerate() {
        if let Some(col) = col {
            evals[row * k + *col as usize] = Fr::from_u64(1);
        }
    }

    let one_hot = OneHotPolynomial::new(k, indices);
    let dense = Polynomial::new(evals);
    let prover_setup = DoryScheme::setup_prover(num_vars);

    let (one_hot_commitment, _) = DoryScheme::commit(&one_hot, &prover_setup).unwrap();
    let (dense_commitment, _) = DoryScheme::commit(dense.evaluations(), &prover_setup).unwrap();

    assert_eq!(
        one_hot_commitment, dense_commitment,
        "one-hot commitment must match the equivalent dense table"
    );
}

#[test]
fn streaming_zk_commitment_is_blinded_and_verifies() {
    let num_vars = 4usize;
    let sigma = num_vars.div_ceil(2);
    let num_cols = 1usize << sigma;
    let mut rng = ChaCha20Rng::seed_from_u64(350);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);

    let mut partial = DoryScheme::begin(&prover_setup);
    for row in poly.evaluations().chunks(num_cols) {
        DoryScheme::feed(&mut partial, row, &prover_setup);
    }
    let (commitment, hint) = DoryScheme::finish_zk(partial, &prover_setup);

    let mut partial_again = DoryScheme::begin(&prover_setup);
    for row in poly.evaluations().chunks(num_cols) {
        DoryScheme::feed(&mut partial_again, row, &prover_setup);
    }
    let (commitment_again, _) = DoryScheme::finish_zk(partial_again, &prover_setup);
    assert_ne!(
        commitment, commitment_again,
        "streaming ZK commitments must use fresh blinding"
    );

    let mut pt = Blake2bTranscript::new(b"stream-zk");
    let (proof, eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();

    let mut vt = Blake2bTranscript::new(b"stream-zk");
    let verified_eval_com =
        DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt)
            .expect("streaming ZK commitment must verify");
    assert_eq!(verified_eval_com, eval_com);
}

#[test]
fn wrong_eval_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(400);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"wrong-eval");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt).unwrap();

    let tampered_eval = eval + Fr::from_u64(1);
    let mut vt = Blake2bTranscript::new(b"wrong-eval");
    let result = DoryScheme::verify(
        &commitment,
        &point,
        tampered_eval,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(result.is_err(), "tampered eval must be rejected");
}

#[test]
fn wrong_point_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(500);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"wrong-point");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt).unwrap();

    let mut tampered_point = point.clone();
    tampered_point[0] += Fr::from_u64(1);
    let mut vt = Blake2bTranscript::new(b"wrong-point");
    let result = DoryScheme::verify(
        &commitment,
        &tampered_point,
        eval,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(result.is_err(), "tampered point must be rejected");
}

#[test]
fn combine_linear_combination() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(600);

    let prover_setup = DoryScheme::setup_prover(num_vars);

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (commit_a, _) = DoryScheme::commit(poly_a.evaluations(), &prover_setup).unwrap();
    let (commit_b, _) = DoryScheme::commit(poly_b.evaluations(), &prover_setup).unwrap();

    let c1 = <Fr as RandomSampling>::random(&mut rng);
    let c2 = <Fr as RandomSampling>::random(&mut rng);

    let combined = DoryScheme::combine(&[commit_a, commit_b], &[c1, c2]);

    let weighted_evals: Vec<Fr> = poly_a
        .evaluations()
        .iter()
        .zip(poly_b.evaluations().iter())
        .map(|(a, b)| c1 * *a + c2 * *b)
        .collect();
    let (commit_weighted, _) = DoryScheme::commit(&weighted_evals, &prover_setup).unwrap();

    assert_eq!(
        combined, commit_weighted,
        "combine must match commitment of weighted sum"
    );
}

#[test]
fn deterministic_commitment() {
    let num_vars = 4;
    let prover_setup = DoryScheme::setup_prover(num_vars);

    let mut rng = ChaCha20Rng::seed_from_u64(700);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (c1, _) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();
    let (c2, _) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

    assert_eq!(c1, c2, "same poly + setup must yield identical commitment");
}

#[test]
fn zk_commitment_is_blinded() {
    let num_vars = 4;
    let prover_setup = DoryScheme::setup_prover(num_vars);

    let mut rng = ChaCha20Rng::seed_from_u64(750);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (c1, _) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();
    let (c2, _) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

    assert_ne!(c1, c2, "ZK Dory commitments must use fresh blinding");
}

#[test]
fn wrong_commitment_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(900);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"wrong-commit");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt).unwrap();

    // Commit to a different polynomial
    let wrong_poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let (wrong_commitment, _) =
        DoryScheme::commit(wrong_poly.evaluations(), &prover_setup).unwrap();
    assert_ne!(commitment, wrong_commitment);

    let mut vt = Blake2bTranscript::new(b"wrong-commit");
    let result = DoryScheme::verify(
        &wrong_commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(result.is_err(), "wrong commitment must be rejected");
}

#[test]
fn wrong_transcript_domain_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1000);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"correct-domain");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt).unwrap();

    let mut vt = Blake2bTranscript::new(b"wrong-domain");
    let result = DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt);
    assert!(result.is_err(), "wrong transcript domain must be rejected");
}

#[test]
fn property_based_round_trip() {
    for seed in 0..10u64 {
        let num_vars = 2 + (seed as usize % 4); // 2..5
        round_trip::<Blake2bTranscript>(num_vars, 800 + seed, b"prop-rt");
    }
}

fn zk_round_trip<T: Transcript<Challenge = Fr>>(num_vars: usize, seed: u64, label: &'static [u8]) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = T::new(label);
    let (proof, eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();

    let mut vt = T::new(label);
    let verified_eval_com =
        DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt)
            .expect("ZK round-trip verification must succeed");
    assert_eq!(verified_eval_com, eval_com);
}

#[test]
fn zk_round_trip_various_sizes() {
    for num_vars in [2, 3, 4, 6] {
        zk_round_trip::<Blake2bTranscript>(num_vars, 1100 + num_vars as u64, b"zk-cov-sizes");
    }
}

#[test]
fn zk_round_trip_both_transcripts() {
    let num_vars = 4;
    zk_round_trip::<Blake2bTranscript>(num_vars, 1200, b"zk-blake2b-rt");
    zk_round_trip::<KeccakTranscript>(num_vars, 1200, b"zk-keccak-rt");
}

#[test]
fn transparent_verify_rejects_zk_opening_proof() {
    let num_vars = 4;
    let mut rng = ChaCha20Rng::seed_from_u64(1301);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"zk-proof-transparent-verify");
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();

    let mut vt = Blake2bTranscript::new(b"zk-proof-transparent-verify");
    let result = DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt);
    assert!(
        result.is_err(),
        "transparent verification must reject ZK opening proofs"
    );
}

#[test]
fn zk_wrong_commitment_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1400);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"zk-wrong-commit");
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();

    let wrong_poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let (wrong_commitment, _) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(wrong_poly.evaluations(), &prover_setup)
            .unwrap();
    assert_ne!(commitment, wrong_commitment);

    let mut vt = Blake2bTranscript::new(b"zk-wrong-commit");
    let result = DoryScheme::verify_zk(&wrong_commitment, &point, &proof, &verifier_setup, &mut vt);
    assert!(result.is_err(), "ZK: wrong commitment must be rejected");
}

#[test]
fn transparent_commitment_rejected_for_zk_blinded_proof() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1450);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (transparent_commitment, _) =
        DoryScheme::commit(poly.evaluations(), &prover_setup).unwrap();
    let (_zk_commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"zk-transparent-reject");
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();

    let mut vt = Blake2bTranscript::new(b"zk-transparent-reject");
    let result = DoryScheme::verify_zk(
        &transparent_commitment,
        &point,
        &proof,
        &verifier_setup,
        &mut vt,
    );
    assert!(
        result.is_err(),
        "transparent commitment must not verify against a proof using a ZK commit blind"
    );
}

#[test]
fn zk_combined_commitment_and_hint_verify() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1475);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let (commit_a, hint_a) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly_a.evaluations(), &prover_setup).unwrap();
    let (commit_b, hint_b) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly_b.evaluations(), &prover_setup).unwrap();

    let c1 = <Fr as RandomSampling>::random(&mut rng);
    let c2 = <Fr as RandomSampling>::random(&mut rng);
    let combined_commitment = DoryScheme::combine(&[commit_a, commit_b], &[c1, c2]);
    let combined_hint = DoryScheme::combine_hints(vec![hint_a, hint_b], &[c1, c2]);

    let weighted_evals: Vec<Fr> = poly_a
        .evaluations()
        .iter()
        .zip(poly_b.evaluations().iter())
        .map(|(a, b)| c1 * *a + c2 * *b)
        .collect();
    let weighted_poly = Polynomial::new(weighted_evals);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = weighted_poly.evaluate(&point);

    let mut pt = Blake2bTranscript::new(b"zk-combined");
    let (proof, eval_com, _blind) = DoryScheme::open_zk(
        &weighted_poly,
        &point,
        eval,
        &prover_setup,
        combined_hint,
        &mut pt,
    )
    .unwrap();

    let mut vt = Blake2bTranscript::new(b"zk-combined");
    let verified_eval_com = DoryScheme::verify_zk(
        &combined_commitment,
        &point,
        &proof,
        &verifier_setup,
        &mut vt,
    )
    .expect("combined ZK commitment and hint must verify");
    assert_eq!(verified_eval_com, eval_com);
}

#[test]
fn wrong_eval_commitment_rejected_zk() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1600);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"zk-tampered-y-com");
    let (mut proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();

    // Replace proof.y_com (the hiding commitment to the evaluation) with a
    // different valid G1. dory::verify must reject because the Σ₁/Σ₂ sub-proofs
    // bind y_com cryptographically to the rest of the proof.
    proof.0.y_com = Some(ArkG1::default());

    let mut vt = Blake2bTranscript::new(b"zk-tampered-y-com");
    let result = DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt);
    assert!(result.is_err(), "tampered proof.y_com must be rejected");
}

#[test]
fn zk_wrong_transcript_domain_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(1500);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup).unwrap();

    let mut pt = Blake2bTranscript::new(b"zk-correct-domain");
    let (proof, _eval_com, _blind) =
        DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut pt).unwrap();

    let mut vt = Blake2bTranscript::new(b"zk-wrong-domain");
    let result = DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt);
    assert!(
        result.is_err(),
        "ZK: wrong transcript domain must be rejected"
    );
}

/// Hints from a shared commitment grid are ragged: a polynomial narrower than
/// the grid streams fewer rows than a grid-spanning one. `combine_hints` pads
/// the narrow hint with identity rows (the zero-embedding's missing rows), so
/// the combined hint must open the RLC of the wide polynomial with the
/// zero-extended narrow one.
#[test]
fn ragged_hint_combination_verifies() {
    let wide_vars = 6;
    let narrow_vars = 4;
    let mut rng = ChaCha20Rng::seed_from_u64(88);
    let prover_setup = DoryScheme::setup_prover(wide_vars);
    let verifier_setup = DoryScheme::setup_verifier(wide_vars);

    let wide = Polynomial::<Fr>::random(wide_vars, &mut rng);
    let narrow = Polynomial::<Fr>::random(narrow_vars, &mut rng);

    let (wide_commit, wide_hint) = DoryScheme::commit(wide.evaluations(), &prover_setup).unwrap();

    // Commit the narrow polynomial at the wide grid's row width, as the
    // shared-grid witness commitment does — fewer hint rows than the wide.
    let row_width = 1usize << wide_vars.div_ceil(2);
    let mut partial = DoryScheme::begin(&prover_setup);
    for chunk in narrow.evaluations().chunks(row_width) {
        DoryScheme::feed(&mut partial, chunk, &prover_setup);
    }
    let (narrow_commit, narrow_hint) = DoryScheme::finish_with_hint(partial, &prover_setup);

    let c1 = <Fr as RandomSampling>::random(&mut rng);
    let c2 = <Fr as RandomSampling>::random(&mut rng);
    let combined_commitment = DoryScheme::combine(&[wide_commit, narrow_commit], &[c1, c2]);
    let combined_hint = DoryScheme::combine_hints(vec![wide_hint, narrow_hint], &[c1, c2]);

    // The joint polynomial: wide + zero-embedded (low-index prefix) narrow.
    let mut narrow_embedded = narrow.evaluations().to_vec();
    narrow_embedded.resize(1 << wide_vars, Fr::from_u64(0));
    let joint = Polynomial::new(
        wide.evaluations()
            .iter()
            .zip(&narrow_embedded)
            .map(|(w, n)| c1 * *w + c2 * *n)
            .collect::<Vec<_>>(),
    );
    let point: Vec<Fr> = (0..wide_vars)
        .map(|_| <Fr as RandomSampling>::random(&mut rng))
        .collect();
    let eval = joint.evaluate(&point);

    let mut pt = Blake2bTranscript::new(b"ragged-hints");
    let proof = DoryScheme::open(
        &joint,
        &point,
        eval,
        &prover_setup,
        Some(combined_hint),
        &mut pt,
    )
    .unwrap();

    let mut vt = Blake2bTranscript::new(b"ragged-hints");
    DoryScheme::verify(
        &combined_commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut vt,
    )
    .expect("ragged hint combination must verify");
}
