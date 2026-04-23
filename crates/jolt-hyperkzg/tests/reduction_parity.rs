//! Parity test for [`HyperKZGScheme::reduce_verifier_with_backend`].
//!
//! Driving the new backend-aware reduction through [`Native`] must produce
//! commitments, points, and evaluations identical to the concrete
//! `reduce_verifier`, and must leave the transcript in the same state.
//! See `jolt-verifier-backend/tests/reduction_parity.rs` for the
//! `MockCommitmentScheme` analogue and the rationale write-up.

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{
    BackendVerifierClaim, CommitmentBackend, CommitmentOrigin, CommitmentScheme, FieldBackend,
    OpeningReduction, ScalarOrigin, VerifierClaim,
};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;
use jolt_verifier_backend::Native;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;
type KzgOutput = <KzgPCS as jolt_openings::Commitment>::Output;

fn make_setup(max_degree: usize) -> (
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

fn make_claims(
    polys: &[Polynomial<Fr>],
    points: &[Vec<Fr>],
    pk: &<KzgPCS as CommitmentScheme>::ProverSetup,
) -> Vec<VerifierClaim<Fr, KzgOutput>> {
    polys
        .iter()
        .zip(points.iter())
        .map(|(poly, point)| {
            let eval = poly.evaluate(point);
            let (commitment, ()) = KzgPCS::commit(poly.evaluations(), pk);
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
    claims: &[VerifierClaim<Fr, KzgOutput>],
) -> Vec<BackendVerifierClaim<Native<Fr>, KzgPCS>> {
    claims
        .iter()
        .map(|c| {
            let commitment = <Native<Fr> as CommitmentBackend<KzgPCS>>::wrap_commitment(
                backend,
                c.commitment,
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

fn check_parity(polys: &[Polynomial<Fr>], points: &[Vec<Fr>], label: &'static [u8]) {
    let max_n = polys
        .iter()
        .map(|p| p.evaluations().len())
        .max()
        .unwrap_or(1);
    let (pk, _vk) = make_setup(max_n);
    let claims = make_claims(polys, points, &pk);

    let mut backend = Native::<Fr>::new();
    let mut transcript_concrete = backend.new_transcript(b"hyperkzg-parity");
    transcript_concrete.append_bytes(label);
    let reduced_concrete = KzgPCS::reduce_verifier(claims.clone(), &mut transcript_concrete)
        .expect("concrete reduce_verifier should succeed");

    let mut transcript_backend = backend.new_transcript(b"hyperkzg-parity");
    transcript_backend.append_bytes(label);
    let backend_claims = lift_to_backend(&mut backend, &claims);
    let reduced_backend =
        KzgPCS::reduce_verifier_with_backend(&mut backend, backend_claims, &mut transcript_backend)
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
    let mut backend = Native::<Fr>::new();
    let mut transcript_concrete = backend.new_transcript(b"hyperkzg-parity");
    let reduced_concrete: Vec<_> =
        KzgPCS::reduce_verifier(Vec::new(), &mut transcript_concrete).unwrap();
    assert!(reduced_concrete.is_empty());

    let mut transcript_backend = backend.new_transcript(b"hyperkzg-parity");
    let reduced_backend: Vec<_> =
        KzgPCS::reduce_verifier_with_backend(&mut backend, Vec::new(), &mut transcript_backend)
            .unwrap();
    assert!(reduced_backend.is_empty());

    let c_concrete: Fr = transcript_concrete.challenge();
    let c_backend: Fr = transcript_backend.challenge();
    assert_eq!(c_concrete, c_backend);
}
