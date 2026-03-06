//! Integration tests for the opening reduction pipeline.
//!
//! These tests exercise the public API only — no internal imports.
//! They verify that the full reduce → open → verify pipeline works
//! end-to-end with MockCommitmentScheme across both transcript backends.
//!
//! Requires: `cargo nextest run -p jolt-openings --features test-utils`

#![cfg(feature = "test-utils")]

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::{
    CommitmentScheme, OpeningReduction, ProverClaim, RlcReduction, VerifierClaim,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;

fn challenge_fn(c: u128) -> Fr {
    Fr::from_u128(c)
}

/// Full reduce → open → verify pipeline.
fn reduce_open_verify<T: Transcript<Challenge = u128>>(
    polys: &[Polynomial<Fr>],
    points: &[Vec<Fr>],
    label: &'static [u8],
) {
    assert_eq!(polys.len(), points.len());

    let mut prover_claims = Vec::new();
    let mut verifier_claims = Vec::new();

    for (poly, point) in polys.iter().zip(points.iter()) {
        let eval = poly.evaluate(point);
        prover_claims.push(ProverClaim {
            evaluations: poly.evaluations().to_vec(),
            point: point.clone(),
            eval,
        });
        verifier_claims.push(VerifierClaim {
            commitment: MockPCS::commit(poly.evaluations(), &()),
            point: point.clone(),
            eval,
        });
    }

    // Prover side
    let mut transcript_p = T::new(label);
    let (reduced_p, ()) = <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(
        prover_claims,
        &mut transcript_p,
        challenge_fn,
    );
    let proofs: Vec<_> = reduced_p
        .iter()
        .map(|c| MockPCS::open(&c.evaluations, &c.point, c.eval, &(), &mut transcript_p))
        .collect();

    // Verifier side
    let mut transcript_v = T::new(label);
    let reduced_v = <RlcReduction as OpeningReduction<MockPCS>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut transcript_v,
        challenge_fn,
    )
    .expect("reduction should succeed");

    assert_eq!(reduced_v.len(), proofs.len());
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        MockPCS::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
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
    reduce_open_verify::<Blake2bTranscript>(&[poly], &[point], b"single-blake2b");
}

#[test]
fn single_claim_keccak() {
    let mut rng = ChaCha20Rng::seed_from_u64(1001);
    let poly = Polynomial::<Fr>::random(4, &mut rng);
    let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    reduce_open_verify::<KeccakTranscript>(&[poly], &[point], b"single-keccak");
}

#[test]
fn multiple_claims_shared_point() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    let nv = 3;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let polys: Vec<_> = (0..5).map(|_| Polynomial::<Fr>::random(nv, &mut rng)).collect();
    let points: Vec<_> = (0..5).map(|_| point.clone()).collect();

    reduce_open_verify::<Blake2bTranscript>(&polys, &points, b"shared-point");
}

#[test]
fn multiple_claims_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(3000);
    let nv = 3;

    let polys: Vec<_> = (0..4).map(|_| Polynomial::<Fr>::random(nv, &mut rng)).collect();
    let points: Vec<Vec<Fr>> = (0..4)
        .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    reduce_open_verify::<Blake2bTranscript>(&polys, &points, b"distinct-points");
}

#[test]
fn mixed_shared_and_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(4000);
    let nv = 4;

    let shared_point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let other_point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let polys: Vec<_> = (0..6).map(|_| Polynomial::<Fr>::random(nv, &mut rng)).collect();
    let points = vec![
        shared_point.clone(),
        shared_point.clone(),
        shared_point,
        other_point.clone(),
        other_point.clone(),
        other_point,
    ];

    reduce_open_verify::<Blake2bTranscript>(&polys, &points, b"mixed-points");
}

#[test]
fn empty_claims_is_noop() {
    let mut transcript_p = Blake2bTranscript::new(b"empty");
    let (reduced, ()) = <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(
        Vec::new(),
        &mut transcript_p,
        challenge_fn,
    );
    assert!(reduced.is_empty());

    let mut transcript_v = Blake2bTranscript::new(b"empty");
    let reduced_v = <RlcReduction as OpeningReduction<MockPCS>>::reduce_verifier(
        Vec::new(),
        &(),
        &mut transcript_v,
        challenge_fn,
    )
    .unwrap();
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
        ProverClaim {
            evaluations: poly_a.evaluations().to_vec(),
            point: point.clone(),
            eval: eval_a,
        },
        ProverClaim {
            evaluations: poly_b.evaluations().to_vec(),
            point: point.clone(),
            eval: eval_b,
        },
    ];

    // Verifier has tampered eval for poly_b
    let verifier_claims = vec![
        VerifierClaim {
            commitment: MockPCS::commit(poly_a.evaluations(), &()),
            point: point.clone(),
            eval: eval_a,
        },
        VerifierClaim {
            commitment: MockPCS::commit(poly_b.evaluations(), &()),
            point: point.clone(),
            eval: eval_b + Fr::from_u64(1), // tampered
        },
    ];

    // Prover reduces and opens honestly
    let mut transcript_p = Blake2bTranscript::new(b"tampered");
    let (reduced_p, ()) = <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(
        prover_claims,
        &mut transcript_p,
        challenge_fn,
    );
    let proofs: Vec<_> = reduced_p
        .iter()
        .map(|c| MockPCS::open(&c.evaluations, &c.point, c.eval, &(), &mut transcript_p))
        .collect();

    // Verifier reduces with tampered claims
    let mut transcript_v = Blake2bTranscript::new(b"tampered");
    let reduced_v = <RlcReduction as OpeningReduction<MockPCS>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut transcript_v,
        challenge_fn,
    )
    .expect("reduction itself should succeed");

    // Verification should fail because combined eval won't match
    let mut any_failed = false;
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        if MockPCS::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            &(),
            &mut transcript_v,
        )
        .is_err()
        {
            any_failed = true;
        }
    }
    assert!(any_failed, "tampered evaluation must cause verification failure");
}

/// Property-based: for any random polynomial count and dimensions,
/// reduce → open → verify succeeds.
#[test]
fn property_random_claims_always_verify() {
    for seed in 6000..6020 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let nv = 2 + (seed as usize % 4); // 2..5 vars
        let num_polys = 1 + (seed as usize % 6); // 1..6 polys
        let num_points = 1 + (seed as usize % 3); // 1..3 distinct points

        let points: Vec<Vec<Fr>> = (0..num_points)
            .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let polys: Vec<_> = (0..num_polys)
            .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
            .collect();

        // Assign each poly to a random point
        let claim_points: Vec<_> = (0..num_polys)
            .map(|i| points[i % num_points].clone())
            .collect();

        reduce_open_verify::<Blake2bTranscript>(
            &polys,
            &claim_points,
            b"property",
        );
    }
}
