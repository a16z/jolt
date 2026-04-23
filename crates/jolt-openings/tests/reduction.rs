//! Integration tests for the fused batched-opening pipeline.
//!
//! Exercises the public `prove_batch` / `verify_batch` API end-to-end
//! with `MockCommitmentScheme` across both transcript backends.
//!
//! Requires: `cargo nextest run -p jolt-openings --features test-utils`

#![cfg(feature = "test-utils")]

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::{
    CommitmentScheme, CommitmentSchemeVerifier, OpeningClaim, OpeningsError, ProverClaim,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;

/// Drive `prove_batch` on prover and `verify_batch` on verifier.
fn prove_verify_batch<T: Transcript<Challenge = Fr>>(
    polys: &[Polynomial<Fr>],
    points: &[Vec<Fr>],
    label: &'static [u8],
) {
    assert_eq!(polys.len(), points.len());

    let mut prover_claims = Vec::new();
    let mut verifier_claims: Vec<OpeningClaim<Fr, MockPCS>> = Vec::new();

    for (poly, point) in polys.iter().zip(points.iter()) {
        let eval = poly.evaluate(point);
        prover_claims.push(ProverClaim {
            polynomial: Polynomial::new(poly.evaluations().to_vec()),
            point: point.clone(),
            eval,
        });
        let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());
        verifier_claims.push(OpeningClaim {
            commitment,
            point: point.clone(),
            eval,
        });
    }

    let hints = vec![(); prover_claims.len()];

    let mut transcript_p = T::new(label);
    let (batch_proof, _joint_evals) =
        MockPCS::prove_batch(prover_claims, hints, &(), &mut transcript_p);

    let mut transcript_v = T::new(label);
    MockPCS::verify_batch(verifier_claims, &batch_proof, &(), &mut transcript_v)
        .expect("batched verification must succeed");
}

#[test]
fn single_claim_blake2b() {
    let mut rng = ChaCha20Rng::seed_from_u64(1000);
    let poly = Polynomial::<Fr>::random(4, &mut rng);
    let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    prove_verify_batch::<Blake2bTranscript>(&[poly], &[point], b"single-blake2b");
}

#[test]
fn single_claim_keccak() {
    let mut rng = ChaCha20Rng::seed_from_u64(1001);
    let poly = Polynomial::<Fr>::random(4, &mut rng);
    let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    prove_verify_batch::<KeccakTranscript>(&[poly], &[point], b"single-keccak");
}

#[test]
fn multiple_claims_shared_point() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    let nv = 3;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let polys: Vec<_> = (0..5)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<_> = (0..5).map(|_| point.clone()).collect();

    prove_verify_batch::<Blake2bTranscript>(&polys, &points, b"shared-point");
}

#[test]
fn multiple_claims_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(3000);
    let nv = 3;

    let polys: Vec<_> = (0..4)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<Vec<Fr>> = (0..4)
        .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    prove_verify_batch::<Blake2bTranscript>(&polys, &points, b"distinct-points");
}

#[test]
fn mixed_shared_and_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(4000);
    let nv = 4;

    let shared_point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let other_point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let polys: Vec<_> = (0..6)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points = vec![
        shared_point.clone(),
        shared_point.clone(),
        shared_point,
        other_point.clone(),
        other_point.clone(),
        other_point,
    ];

    prove_verify_batch::<Blake2bTranscript>(&polys, &points, b"mixed-points");
}

#[test]
fn empty_claims_is_noop() {
    let mut transcript_p = Blake2bTranscript::new(b"empty");
    let (batch_proof, joint_evals) =
        MockPCS::prove_batch(Vec::new(), Vec::new(), &(), &mut transcript_p);
    assert!(batch_proof.is_empty());
    assert!(joint_evals.is_empty());

    let mut transcript_v = Blake2bTranscript::new(b"empty");
    MockPCS::verify_batch(Vec::new(), &batch_proof, &(), &mut transcript_v)
        .expect("empty batch must verify");
}

#[test]
fn tampered_eval_detected() {
    let mut rng = ChaCha20Rng::seed_from_u64(5000);
    let nv = 3;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let poly_a = Polynomial::<Fr>::random(nv, &mut rng);
    let poly_b = Polynomial::<Fr>::random(nv, &mut rng);

    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);

    let prover_claims = vec![
        ProverClaim {
            polynomial: Polynomial::new(poly_a.evaluations().to_vec()),
            point: point.clone(),
            eval: eval_a,
        },
        ProverClaim {
            polynomial: Polynomial::new(poly_b.evaluations().to_vec()),
            point: point.clone(),
            eval: eval_b,
        },
    ];
    let hints = vec![(); prover_claims.len()];

    let (com_a, ()) = MockPCS::commit(poly_a.evaluations(), &());
    let (com_b, ()) = MockPCS::commit(poly_b.evaluations(), &());
    let verifier_claims: Vec<OpeningClaim<Fr, MockPCS>> = vec![
        OpeningClaim {
            commitment: com_a,
            point: point.clone(),
            eval: eval_a,
        },
        OpeningClaim {
            commitment: com_b,
            point: point.clone(),
            eval: eval_b + Fr::from_u64(1), // tampered
        },
    ];

    let mut transcript_p = Blake2bTranscript::new(b"tampered");
    let (batch_proof, _) = MockPCS::prove_batch(prover_claims, hints, &(), &mut transcript_p);

    let mut transcript_v = Blake2bTranscript::new(b"tampered");
    let result = MockPCS::verify_batch(verifier_claims, &batch_proof, &(), &mut transcript_v);
    // Any error is acceptable — the tampered eval is mixed into the transcript
    // before the RLC challenge, which usually surfaces as `CommitmentMismatch`
    // (combined commitments differ) but could also be `VerificationFailed`
    // (combined evals differ at the same RLC challenge).
    assert!(
        matches!(
            result,
            Err(OpeningsError::VerificationFailed | OpeningsError::CommitmentMismatch { .. })
        ),
        "tampered eval must be detected, got {result:?}"
    );
}

#[test]
fn property_random_claims_always_verify() {
    for seed in 6000..6020 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let nv = 2 + (seed as usize % 4);
        let num_polys = 1 + (seed as usize % 6);
        let num_points = 1 + (seed as usize % 3);

        let points: Vec<Vec<Fr>> = (0..num_points)
            .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let polys: Vec<_> = (0..num_polys)
            .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
            .collect();

        let claim_points: Vec<_> = (0..num_polys)
            .map(|i| points[i % num_points].clone())
            .collect();

        prove_verify_batch::<Blake2bTranscript>(&polys, &claim_points, b"property");
    }
}
