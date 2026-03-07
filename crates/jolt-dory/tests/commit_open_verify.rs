//! Integration tests for the Dory commitment scheme.
//!
//! Public-API-only tests — no `pub(crate)` imports. Exercises commit, open,
//! verify, streaming, combine, and negative cases across transcript backends.

use jolt_dory::DoryScheme;
use jolt_field::{Field, Fr};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, StreamingCommitment};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn round_trip<T: Transcript>(num_vars: usize, seed: u64, label: &'static [u8]) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as Field>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    // With hint
    let mut pt = T::new(label);
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

    let mut vt = T::new(label);
    DoryScheme::verify(&commitment, &point, eval, &proof, &verifier_setup, &mut vt)
        .expect("round-trip verification (with hint) must succeed");

    // Without hint
    let mut pt2 = T::new(label);
    let proof2 = DoryScheme::open(&poly, &point, eval, &prover_setup, None, &mut pt2);

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

        let (direct, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);

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
fn wrong_eval_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(400);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as Field>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"wrong-eval");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

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
        .map(|_| <Fr as Field>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"wrong-point");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

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

    let (commit_a, _) = DoryScheme::commit(poly_a.evaluations(), &prover_setup);
    let (commit_b, _) = DoryScheme::commit(poly_b.evaluations(), &prover_setup);

    let c1 = <Fr as Field>::random(&mut rng);
    let c2 = <Fr as Field>::random(&mut rng);

    let combined = DoryScheme::combine(&[commit_a, commit_b], &[c1, c2]);

    let weighted_evals: Vec<Fr> = poly_a
        .evaluations()
        .iter()
        .zip(poly_b.evaluations().iter())
        .map(|(a, b)| c1 * *a + c2 * *b)
        .collect();
    let (commit_weighted, _) = DoryScheme::commit(&weighted_evals, &prover_setup);

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

    let (c1, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);
    let (c2, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    assert_eq!(c1, c2, "same poly + setup must yield identical commitment");
}

#[test]
fn wrong_commitment_rejected() {
    let num_vars = 3;
    let mut rng = ChaCha20Rng::seed_from_u64(900);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars)
        .map(|_| <Fr as Field>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"wrong-commit");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

    // Commit to a different polynomial
    let wrong_poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let (wrong_commitment, _) = DoryScheme::commit(wrong_poly.evaluations(), &prover_setup);
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
        .map(|_| <Fr as Field>::random(&mut rng))
        .collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

    let mut pt = Blake2bTranscript::new(b"correct-domain");
    let proof = DoryScheme::open(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

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
