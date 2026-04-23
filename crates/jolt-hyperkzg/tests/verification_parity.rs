//! Parity test for [`HyperKZGScheme::verify_batch_with_backend`].
//!
//! Driving `prove_batch` then `verify_batch_with_backend` (with [`Native`])
//! must accept the proof and leave both transcripts in byte-identical
//! state. See `jolt-verifier-backend/tests/verification_parity.rs` for
//! the `MockCommitmentScheme` analogue and the rationale write-up.

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{
    CommitmentBackend, CommitmentOrigin, CommitmentScheme, FieldBackend, OpeningClaim,
    OpeningVerification, ProverClaim, ScalarOrigin,
};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;
use jolt_verifier_backend::Native;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;
type KzgOutput = <KzgPCS as jolt_openings::Commitment>::Output;
type KzgHint = <KzgPCS as CommitmentScheme>::OpeningHint;

fn make_setup(
    max_degree: usize,
) -> (
    <KzgPCS as CommitmentScheme>::ProverSetup,
    <KzgPCS as CommitmentScheme>::VerifierSetup,
) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xf00d_d00d);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = KzgPCS::setup(&mut rng, max_degree, g1, g2);
    let vk = KzgPCS::verifier_setup(&pk);
    (pk, vk)
}

struct ClaimSet {
    prover: Vec<ProverClaim<Fr>>,
    hints: Vec<KzgHint>,
    verifier_native: Vec<(KzgOutput, Vec<Fr>, Fr)>,
}

fn make_claims(
    polys: &[Polynomial<Fr>],
    points: &[Vec<Fr>],
    pk: &<KzgPCS as CommitmentScheme>::ProverSetup,
) -> ClaimSet {
    let mut prover = Vec::with_capacity(polys.len());
    let mut hints = Vec::with_capacity(polys.len());
    let mut verifier_native = Vec::with_capacity(polys.len());
    for (poly, point) in polys.iter().zip(points.iter()) {
        let eval = poly.evaluate(point);
        let (commitment, hint) = KzgPCS::commit(poly.evaluations(), pk);
        prover.push(ProverClaim {
            polynomial: poly.clone(),
            point: point.clone(),
            eval,
        });
        hints.push(hint);
        verifier_native.push((commitment, point.clone(), eval));
    }
    ClaimSet {
        prover,
        hints,
        verifier_native,
    }
}

fn lift_to_backend(
    backend: &mut Native<Fr>,
    claims: &[(KzgOutput, Vec<Fr>, Fr)],
) -> Vec<OpeningClaim<Native<Fr>, KzgPCS>> {
    claims
        .iter()
        .map(|(commitment, point, eval)| {
            let commitment_handle = <Native<Fr> as CommitmentBackend<KzgPCS>>::wrap_commitment(
                backend,
                *commitment,
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

fn check_parity(polys: &[Polynomial<Fr>], points: &[Vec<Fr>], label: &'static [u8]) {
    let max_n = polys
        .iter()
        .map(|p| p.evaluations().len())
        .max()
        .unwrap_or(1);
    let (pk, vk) = make_setup(max_n);
    let claim_set = make_claims(polys, points, &pk);

    let mut backend = Native::<Fr>::new();

    let mut prover_transcript = backend.new_transcript(b"hyperkzg-parity");
    prover_transcript.append_bytes(label);
    let (batch_proof, _binding_evals) = KzgPCS::prove_batch(
        claim_set.prover,
        claim_set.hints,
        &pk,
        &mut prover_transcript,
    );

    let mut verifier_transcript = backend.new_transcript(b"hyperkzg-parity");
    verifier_transcript.append_bytes(label);
    let backend_claims = lift_to_backend(&mut backend, &claim_set.verifier_native);
    KzgPCS::verify_batch_with_backend(
        &mut backend,
        &vk,
        backend_claims,
        &batch_proof,
        &mut verifier_transcript,
    )
    .expect("verify_batch_with_backend should accept a valid HyperKZG batch proof");

    let challenge_prover: Fr = prover_transcript.challenge();
    let challenge_verifier: Fr = verifier_transcript.challenge();
    assert_eq!(
        challenge_prover, challenge_verifier,
        "prover and verifier transcripts must be in the same state after the batched opening",
    );
}

#[test]
fn hyperkzg_parity_single_claim() {
    let mut rng = ChaCha20Rng::seed_from_u64(7000);
    let nv = 4;
    let poly = Polynomial::<Fr>::random(nv, &mut rng);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    check_parity(&[poly], &[point], b"single");
}

#[test]
fn hyperkzg_parity_shared_point_batch() {
    let mut rng = ChaCha20Rng::seed_from_u64(7001);
    let nv = 3;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let polys: Vec<_> = (0..4)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<_> = (0..4).map(|_| point.clone()).collect();
    check_parity(&polys, &points, b"shared");
}

#[test]
fn hyperkzg_parity_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(7002);
    let nv = 3;
    let polys: Vec<_> = (0..3)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<Vec<Fr>> = (0..3)
        .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
        .collect();
    check_parity(&polys, &points, b"distinct");
}

#[test]
fn hyperkzg_parity_mixed_groups() {
    let mut rng = ChaCha20Rng::seed_from_u64(7003);
    let nv = 4;
    let shared: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let other: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let polys: Vec<_> = (0..4)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points = vec![shared.clone(), shared, other.clone(), other];
    check_parity(&polys, &points, b"mixed");
}

#[test]
fn hyperkzg_parity_empty_is_noop() {
    let (_pk, vk) = make_setup(1);
    let mut backend = Native::<Fr>::new();

    let mut prover_transcript = backend.new_transcript(b"hyperkzg-parity");
    let (batch_proof, binding_evals) =
        KzgPCS::prove_batch(Vec::new(), Vec::new(), &_pk, &mut prover_transcript);
    assert!(batch_proof.is_empty());
    assert!(binding_evals.is_empty());

    let mut verifier_transcript = backend.new_transcript(b"hyperkzg-parity");
    KzgPCS::verify_batch_with_backend(
        &mut backend,
        &vk,
        Vec::<OpeningClaim<Native<Fr>, KzgPCS>>::new(),
        &batch_proof,
        &mut verifier_transcript,
    )
    .expect("empty batch must verify trivially");

    let c_prover: Fr = prover_transcript.challenge();
    let c_verifier: Fr = verifier_transcript.challenge();
    assert_eq!(c_prover, c_verifier);
}
