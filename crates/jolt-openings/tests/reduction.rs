//! Integration tests for the opening reduction pipeline.
//!
//! These tests exercise the public API only — no internal imports.
//! They verify that the full reduce → open → verify pipeline works
//! end-to-end with MockCommitmentScheme across both transcript backends.
//!
//! Requires: `cargo nextest run -p jolt-openings --features test-utils`

#![cfg(feature = "test-utils")]
#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests may panic on assertion failures"
)]

use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::{
    reduce_prover, reduce_verifier, CommitmentScheme, EvaluationClaim, ProverOpeningClaim,
    VerifierOpeningClaim,
};
use jolt_poly::Polynomial;
use jolt_transcript::{
    prover_transcript, verifier_transcript, Blake2b512, DuplexSpongeInterface, Keccak, ProverState,
    VerifierState,
};
use rand::rngs::StdRng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;

const INSTANCE: [u8; 32] = [0u8; 32];

/// Full reduce → open → verify pipeline. Generic over the spongefish sponge:
/// the modular crates are symmetric (both sides `absorb`; no NARG), so the
/// prover uses a `ProverState` and the verifier an independently-built
/// `VerifierState` over an empty NARG, and they derive identical challenges.
fn reduce_open_verify<H>(polys: &[Polynomial<Fr>], points: &[Vec<Fr>], label: &'static [u8])
where
    H: DuplexSpongeInterface<U = u8> + Default,
    ProverState<H, StdRng>: jolt_transcript::FsTranscript<Fr>,
    for<'a> VerifierState<'a, H>: jolt_transcript::FsTranscript<Fr>,
{
    assert_eq!(polys.len(), points.len());

    let mut prover_claims = Vec::new();
    let mut verifier_claims = Vec::new();

    for (poly, point) in polys.iter().zip(points.iter()) {
        let eval = poly.evaluate(point);
        prover_claims.push(ProverOpeningClaim {
            polynomial: Polynomial::new(poly.evaluations().to_vec()),
            evaluation: EvaluationClaim::new(point.clone(), eval),
        });
        let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());
        verifier_claims.push(VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), eval),
        });
    }

    // Prover side
    let mut transcript_p = prover_transcript(label, INSTANCE, H::default());
    let reduced_p = reduce_prover(prover_claims, &mut transcript_p);
    let proofs: Vec<_> = reduced_p
        .iter()
        .map(|c| {
            MockPCS::open(
                &c.polynomial,
                &c.evaluation.point,
                c.evaluation.value,
                &(),
                None,
                &mut transcript_p,
            )
        })
        .collect();

    // Verifier side
    let mut transcript_v = verifier_transcript(label, INSTANCE, H::default(), &[]);
    let reduced_v = reduce_verifier::<MockPCS, _>(verifier_claims, &mut transcript_v)
        .expect("reduction should succeed");

    assert_eq!(reduced_v.len(), proofs.len());
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        MockPCS::verify(
            &claim.commitment,
            &claim.evaluation.point,
            claim.evaluation.value,
            proof,
            &(),
            &mut transcript_v,
        )
        .expect("verification should succeed");
    }
}

#[test]
fn single_claim_blake2b() {
    let mut rng = ChaCha20Rng::seed_from_u64(1000);
    let poly = Polynomial::<Fr>::random(4, &mut rng);
    let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    reduce_open_verify::<Blake2b512>(&[poly], &[point], b"single-blake2b");
}

#[test]
fn single_claim_keccak() {
    let mut rng = ChaCha20Rng::seed_from_u64(1001);
    let poly = Polynomial::<Fr>::random(4, &mut rng);
    let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    reduce_open_verify::<Keccak>(&[poly], &[point], b"single-keccak");
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

    reduce_open_verify::<Blake2b512>(&polys, &points, b"shared-point");
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

    reduce_open_verify::<Blake2b512>(&polys, &points, b"distinct-points");
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

    reduce_open_verify::<Blake2b512>(&polys, &points, b"mixed-points");
}

#[test]
fn empty_claims_is_noop() {
    let mut transcript_p = prover_transcript(b"empty", INSTANCE, Blake2b512::default());
    let reduced = reduce_prover::<Fr, _>(Vec::new(), &mut transcript_p);
    assert!(reduced.is_empty());

    let mut transcript_v = verifier_transcript(b"empty", INSTANCE, Blake2b512::default(), &[]);
    let reduced_v = reduce_verifier::<MockPCS, _>(Vec::new(), &mut transcript_v).unwrap();
    assert!(reduced_v.is_empty());
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
        ProverOpeningClaim {
            polynomial: Polynomial::new(poly_a.evaluations().to_vec()),
            evaluation: EvaluationClaim::new(point.clone(), eval_a),
        },
        ProverOpeningClaim {
            polynomial: Polynomial::new(poly_b.evaluations().to_vec()),
            evaluation: EvaluationClaim::new(point.clone(), eval_b),
        },
    ];

    let (com_a, ()) = MockPCS::commit(poly_a.evaluations(), &());
    let (com_b, ()) = MockPCS::commit(poly_b.evaluations(), &());
    let verifier_claims = vec![
        VerifierOpeningClaim {
            commitment: com_a,
            evaluation: EvaluationClaim::new(point.clone(), eval_a),
        },
        VerifierOpeningClaim {
            commitment: com_b,
            evaluation: EvaluationClaim::new(point.clone(), eval_b + Fr::from_u64(1)),
        },
    ];

    let mut transcript_p = prover_transcript(b"tampered", INSTANCE, Blake2b512::default());
    let reduced_p = reduce_prover(prover_claims, &mut transcript_p);
    let proofs: Vec<_> = reduced_p
        .iter()
        .map(|c| {
            MockPCS::open(
                &c.polynomial,
                &c.evaluation.point,
                c.evaluation.value,
                &(),
                None,
                &mut transcript_p,
            )
        })
        .collect();

    let mut transcript_v = verifier_transcript(b"tampered", INSTANCE, Blake2b512::default(), &[]);
    let reduced_v = reduce_verifier::<MockPCS, _>(verifier_claims, &mut transcript_v)
        .expect("reduction itself should succeed");

    let mut any_failed = false;
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        if MockPCS::verify(
            &claim.commitment,
            &claim.evaluation.point,
            claim.evaluation.value,
            proof,
            &(),
            &mut transcript_v,
        )
        .is_err()
        {
            any_failed = true;
        }
    }
    assert!(
        any_failed,
        "tampered evaluation must cause verification failure"
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

        reduce_open_verify::<Blake2b512>(&polys, &claim_points, b"property");
    }
}
