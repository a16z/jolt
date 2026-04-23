//! Per-PCS parity tests for [`OpeningReduction::reduce_verifier_with_backend`].
//!
//! For every homomorphic PCS (currently [`MockCommitmentScheme`]; the
//! HyperKZG / Dory variants live alongside their own crates), driving
//! `reduce_verifier_with_backend` through [`Native`] must produce a
//! reduction that is byte-identical to the existing concrete
//! `reduce_verifier`:
//!
//! 1. The same number of reduced claims.
//! 2. Identical commitments (Native's `Commitment = PCS::Output`).
//! 3. Identical points and evaluations.
//! 4. The transcript state after reduction (squeezed challenge) matches
//!    bit-for-bit, so a follow-up sumcheck or PCS verify call sees the
//!    exact same Fiat-Shamir state regardless of which entry point
//!    drove the reduction.
//!
//! This is the fastest signal that the new backend-aware reduction
//! preserves the protocol's transcript discipline. The deeper
//! end-to-end check (BlindFold + verify_with_backend) lives in
//! `jolt-equivalence`'s `modular_self_verify_via_tracing_backend` test.

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::{
    BackendVerifierClaim, Commitment, CommitmentBackend, CommitmentOrigin, CommitmentScheme,
    FieldBackend, OpeningReduction, ScalarOrigin, VerifierClaim,
};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;
use jolt_verifier_backend::Native;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;
type MockOutput = <MockPCS as Commitment>::Output;

fn make_claims(
    polys: &[Polynomial<Fr>],
    points: &[Vec<Fr>],
) -> Vec<VerifierClaim<Fr, MockOutput>> {
    polys
        .iter()
        .zip(points.iter())
        .map(|(poly, point)| {
            let eval = poly.evaluate(point);
            let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());
            VerifierClaim {
                commitment,
                point: point.clone(),
                eval,
            }
        })
        .collect()
}

fn lift_to_backend(
    backend: &mut Native<Fr>,
    claims: &[VerifierClaim<Fr, MockOutput>],
) -> Vec<BackendVerifierClaim<Native<Fr>, MockPCS>> {
    claims
        .iter()
        .map(|c| {
            let commitment = <Native<Fr> as CommitmentBackend<MockPCS>>::wrap_commitment(
                backend,
                c.commitment.clone(),
                CommitmentOrigin::Proof,
                "parity_commitment",
            );
            let point: Vec<Fr> = c
                .point
                .iter()
                .map(|p| backend.wrap(*p, ScalarOrigin::Challenge, "parity_point"))
                .collect();
            let eval = backend.wrap(c.eval, ScalarOrigin::Proof, "parity_eval");
            (commitment, point, eval)
        })
        .collect()
}

/// Two reductions must produce bit-identical (commitment, point, eval)
/// triples and leave the transcript in the same state. Native's
/// commitment / scalar types are zero-cost identity wraps over the
/// concrete `PCS::Output` and `Fr`, so the comparison is trivial.
fn check_parity(polys: &[Polynomial<Fr>], points: &[Vec<Fr>], label: &'static [u8]) {
    let claims = make_claims(polys, points);

    let mut backend = Native::<Fr>::new();
    let mut transcript_concrete = backend.new_transcript(b"parity-test");
    transcript_concrete.append_bytes(label);
    let reduced_concrete = MockPCS::reduce_verifier(claims.clone(), &mut transcript_concrete)
        .expect("concrete reduce_verifier should succeed");

    let mut transcript_backend = backend.new_transcript(b"parity-test");
    transcript_backend.append_bytes(label);
    let backend_claims = lift_to_backend(&mut backend, &claims);
    let reduced_backend =
        MockPCS::reduce_verifier_with_backend(&mut backend, backend_claims, &mut transcript_backend)
            .expect("backend reduce_verifier should succeed");

    assert_eq!(
        reduced_concrete.len(),
        reduced_backend.len(),
        "reduce_verifier_with_backend must produce the same number of grouped claims",
    );
    for (idx, (concrete, backend_claim)) in reduced_concrete
        .iter()
        .zip(reduced_backend.iter())
        .enumerate()
    {
        assert_eq!(
            concrete.commitment, backend_claim.0,
            "commitment mismatch at group {idx}",
        );
        assert_eq!(
            concrete.point, backend_claim.1,
            "point mismatch at group {idx}",
        );
        assert_eq!(
            concrete.eval, backend_claim.2,
            "evaluation mismatch at group {idx}",
        );
    }

    let challenge_concrete: Fr = transcript_concrete.challenge();
    let challenge_backend: Fr = transcript_backend.challenge();
    assert_eq!(
        challenge_concrete, challenge_backend,
        "transcript must be in the same state after both reductions",
    );
}

#[test]
fn mock_parity_single_claim() {
    let mut rng = ChaCha20Rng::seed_from_u64(9000);
    let nv = 4;
    let poly = Polynomial::<Fr>::random(nv, &mut rng);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    check_parity(&[poly], &[point], b"single");
}

#[test]
fn mock_parity_shared_point_batch() {
    let mut rng = ChaCha20Rng::seed_from_u64(9001);
    let nv = 3;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let polys: Vec<_> = (0..5)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<_> = (0..5).map(|_| point.clone()).collect();
    check_parity(&polys, &points, b"shared");
}

#[test]
fn mock_parity_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(9002);
    let nv = 3;
    let polys: Vec<_> = (0..4)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<Vec<Fr>> = (0..4)
        .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
        .collect();
    check_parity(&polys, &points, b"distinct");
}

#[test]
fn mock_parity_mixed_groups() {
    let mut rng = ChaCha20Rng::seed_from_u64(9003);
    let nv = 4;
    let shared: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let other: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let polys: Vec<_> = (0..6)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points = vec![
        shared.clone(),
        shared.clone(),
        shared,
        other.clone(),
        other.clone(),
        other,
    ];
    check_parity(&polys, &points, b"mixed");
}

#[test]
fn mock_parity_empty_is_noop() {
    let mut backend = Native::<Fr>::new();
    let mut transcript_concrete = backend.new_transcript(b"parity-test");
    let reduced_concrete: Vec<_> =
        MockPCS::reduce_verifier(Vec::new(), &mut transcript_concrete).unwrap();
    assert!(reduced_concrete.is_empty());

    let mut transcript_backend = backend.new_transcript(b"parity-test");
    let reduced_backend: Vec<_> =
        MockPCS::reduce_verifier_with_backend(&mut backend, Vec::new(), &mut transcript_backend)
            .unwrap();
    assert!(reduced_backend.is_empty());

    // Both transcripts must be untouched by the no-op path.
    let c_concrete: Fr = transcript_concrete.challenge();
    let c_backend: Fr = transcript_backend.challenge();
    assert_eq!(c_concrete, c_backend);
}
