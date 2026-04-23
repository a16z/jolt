//! Per-PCS parity tests for [`OpeningVerification::verify_batch_with_backend`].
//!
//! For every PCS-implements-`OpeningVerification` (currently the
//! [`MockCommitmentScheme`]; the HyperKZG / Dory variants live alongside
//! their own crates), we check that:
//!
//! 1. Driving `prove_batch` then `verify_batch_with_backend` (with
//!    `Native`) accepts the proof.
//! 2. The prover and verifier transcripts end in the same byte-identical
//!    state — witnessed by squeezing one final challenge on each side
//!    and asserting equality.
//!
//! These are the fastest signal that the new fused batch surface
//! preserves the protocol's transcript discipline. The deeper
//! end-to-end check (BlindFold + verify_with_backend) lives in
//! `jolt-equivalence`'s `modular_self_verify_via_tracing_backend` test.

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::{
    Commitment, CommitmentBackend, CommitmentOrigin, CommitmentScheme, FieldBackend, OpeningClaim,
    OpeningVerification, ProverClaim, ScalarOrigin,
};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;
use jolt_verifier_backend::Native;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;
type MockOutput = <MockPCS as Commitment>::Output;

struct ClaimSet {
    prover: Vec<ProverClaim<Fr>>,
    verifier_native: Vec<(MockOutput, Vec<Fr>, Fr)>,
}

fn make_claims(polys: &[Polynomial<Fr>], points: &[Vec<Fr>]) -> ClaimSet {
    let mut prover = Vec::with_capacity(polys.len());
    let mut verifier_native = Vec::with_capacity(polys.len());
    for (poly, point) in polys.iter().zip(points.iter()) {
        let eval = poly.evaluate(point);
        let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());
        prover.push(ProverClaim {
            polynomial: poly.clone(),
            point: point.clone(),
            eval,
        });
        verifier_native.push((commitment, point.clone(), eval));
    }
    ClaimSet {
        prover,
        verifier_native,
    }
}

fn lift_to_backend(
    backend: &mut Native<Fr>,
    claims: &[(MockOutput, Vec<Fr>, Fr)],
) -> Vec<OpeningClaim<Native<Fr>, MockPCS>> {
    claims
        .iter()
        .map(|(commitment, point, eval)| {
            let commitment_handle = <Native<Fr> as CommitmentBackend<MockPCS>>::wrap_commitment(
                backend,
                commitment.clone(),
                CommitmentOrigin::Proof,
                "parity_commitment",
            );
            let point_handles: Vec<Fr> = point
                .iter()
                .map(|p| backend.wrap(*p, ScalarOrigin::Challenge, "parity_point"))
                .collect();
            let eval_handle = backend.wrap(*eval, ScalarOrigin::Proof, "parity_eval");
            OpeningClaim {
                commitment: commitment_handle,
                point: point_handles,
                eval: eval_handle,
            }
        })
        .collect()
}

/// `prove_batch` then `verify_batch_with_backend` must accept and end on
/// byte-identical transcript state.
fn check_parity(polys: &[Polynomial<Fr>], points: &[Vec<Fr>], label: &'static [u8]) {
    let claim_set = make_claims(polys, points);
    let hints: Vec<()> = vec![(); polys.len()];

    let mut backend = Native::<Fr>::new();

    let mut prover_transcript = backend.new_transcript(b"parity-test");
    prover_transcript.append_bytes(label);
    let (batch_proof, _binding_evals) =
        MockPCS::prove_batch(claim_set.prover, hints, &(), &mut prover_transcript);

    let mut verifier_transcript = backend.new_transcript(b"parity-test");
    verifier_transcript.append_bytes(label);
    let backend_claims = lift_to_backend(&mut backend, &claim_set.verifier_native);
    MockPCS::verify_batch_with_backend(
        &mut backend,
        &(),
        backend_claims,
        &batch_proof,
        &mut verifier_transcript,
    )
    .expect("verify_batch_with_backend should accept a valid batch proof");

    let challenge_prover: Fr = prover_transcript.challenge();
    let challenge_verifier: Fr = verifier_transcript.challenge();
    assert_eq!(
        challenge_prover, challenge_verifier,
        "prover and verifier transcripts must be in the same state after the batched opening",
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

    let mut prover_transcript = backend.new_transcript(b"parity-test");
    let (batch_proof, binding_evals) =
        MockPCS::prove_batch(Vec::new(), Vec::new(), &(), &mut prover_transcript);
    assert!(batch_proof.is_empty());
    assert!(binding_evals.is_empty());

    let mut verifier_transcript = backend.new_transcript(b"parity-test");
    MockPCS::verify_batch_with_backend(
        &mut backend,
        &(),
        Vec::<OpeningClaim<Native<Fr>, MockPCS>>::new(),
        &batch_proof,
        &mut verifier_transcript,
    )
    .expect("empty batch must verify trivially");

    let c_prover: Fr = prover_transcript.challenge();
    let c_verifier: Fr = verifier_transcript.challenge();
    assert_eq!(c_prover, c_verifier);
}
