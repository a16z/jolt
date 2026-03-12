//! RLC reduction integration tests with HyperKZG (BN254).
//!
//! Exercises the full reduce → open → verify pipeline using real pairing-based
//! polynomial commitments instead of MockPCS.
//!
//! Requires: `cargo nextest run -p jolt-openings --features test-utils`

#![cfg(feature = "test-utils")]

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{CommitmentScheme, OpeningReduction, ProverClaim, RlcReduction, VerifierClaim};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;

fn make_setup(max_degree: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = KzgPCS::setup(&mut rng, max_degree, g1, g2);
    let vk = KzgPCS::verifier_setup(&pk);
    (pk, vk)
}

/// Full reduce → open → verify pipeline with HyperKZG.
fn reduce_open_verify(
    polys: &[Polynomial<Fr>],
    points: &[Vec<Fr>],
    pk: &HyperKZGProverSetup<Bn254>,
    vk: &HyperKZGVerifierSetup<Bn254>,
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
        let (commitment, ()) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), pk);
        verifier_claims.push(VerifierClaim {
            commitment,
            point: point.clone(),
            eval,
        });
    }

    // Prover: reduce + open
    let mut transcript_p = Blake2bTranscript::new(label);
    let (reduced_p, ()) =
        <RlcReduction as OpeningReduction<KzgPCS>>::reduce_prover(prover_claims, &mut transcript_p);
    let proofs: Vec<_> = reduced_p
        .iter()
        .map(|c| {
            let poly: Polynomial<Fr> = c.evaluations.clone().into();
            <KzgPCS as CommitmentScheme>::open(&poly, &c.point, c.eval, pk, None, &mut transcript_p)
        })
        .collect();

    // Verifier: reduce + verify
    let mut transcript_v = Blake2bTranscript::new(label);
    let reduced_v = <RlcReduction as OpeningReduction<KzgPCS>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut transcript_v,
    )
    .expect("reduction should succeed");

    assert_eq!(reduced_v.len(), proofs.len());
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        <KzgPCS as CommitmentScheme>::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            vk,
            &mut transcript_v,
        )
        .expect("verification should succeed");
    }
}

#[test]
fn single_claim() {
    let mut rng = ChaCha20Rng::seed_from_u64(7000);
    let nv = 4;
    let (pk, vk) = make_setup(1 << nv);

    let poly = Polynomial::<Fr>::random(nv, &mut rng);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    reduce_open_verify(&[poly], &[point], &pk, &vk, b"kzg-single");
}

#[test]
fn multiple_claims_shared_point() {
    let mut rng = ChaCha20Rng::seed_from_u64(7001);
    let nv = 3;
    let (pk, vk) = make_setup(1 << nv);

    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let polys: Vec<_> = (0..5)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<_> = (0..5).map(|_| point.clone()).collect();

    reduce_open_verify(&polys, &points, &pk, &vk, b"kzg-shared");
}

#[test]
fn multiple_claims_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(7002);
    let nv = 3;
    let (pk, vk) = make_setup(1 << nv);

    let polys: Vec<_> = (0..4)
        .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
        .collect();
    let points: Vec<Vec<Fr>> = (0..4)
        .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    reduce_open_verify(&polys, &points, &pk, &vk, b"kzg-distinct");
}

#[test]
fn mixed_shared_and_distinct_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(7003);
    let nv = 4;
    let (pk, vk) = make_setup(1 << nv);

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

    reduce_open_verify(&polys, &points, &pk, &vk, b"kzg-mixed");
}

/// Prover opens honestly, but verifier has a tampered evaluation.
/// The PCS verify must detect the mismatch.
#[test]
fn tampered_eval_detected() {
    let mut rng = ChaCha20Rng::seed_from_u64(7004);
    let nv = 3;
    let (pk, vk) = make_setup(1 << nv);

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

    let (com_a, ()) = <KzgPCS as CommitmentScheme>::commit(poly_a.evaluations(), &pk);
    let (com_b, ()) = <KzgPCS as CommitmentScheme>::commit(poly_b.evaluations(), &pk);
    let verifier_claims = vec![
        VerifierClaim {
            commitment: com_a,
            point: point.clone(),
            eval: eval_a,
        },
        VerifierClaim {
            commitment: com_b,
            point: point.clone(),
            eval: eval_b + Fr::from_u64(1), // tampered
        },
    ];

    // Prover reduces and opens honestly
    let mut transcript_p = Blake2bTranscript::new(b"kzg-tampered");
    let (reduced_p, ()) =
        <RlcReduction as OpeningReduction<KzgPCS>>::reduce_prover(prover_claims, &mut transcript_p);
    let proofs: Vec<_> = reduced_p
        .iter()
        .map(|c| {
            let poly: Polynomial<Fr> = c.evaluations.clone().into();
            <KzgPCS as CommitmentScheme>::open(
                &poly,
                &c.point,
                c.eval,
                &pk,
                None,
                &mut transcript_p,
            )
        })
        .collect();

    // Verifier reduces with tampered claims
    let mut transcript_v = Blake2bTranscript::new(b"kzg-tampered");
    let reduced_v = <RlcReduction as OpeningReduction<KzgPCS>>::reduce_verifier(
        verifier_claims,
        &(),
        &mut transcript_v,
    )
    .expect("reduction itself should succeed");

    let mut any_failed = false;
    for (claim, proof) in reduced_v.iter().zip(proofs.iter()) {
        if <KzgPCS as CommitmentScheme>::verify(
            &claim.commitment,
            &claim.point,
            claim.eval,
            proof,
            &vk,
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

/// Property test: random claim counts and dimensions always verify.
#[test]
fn property_random_claims_always_verify() {
    for seed in 8000..8010 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let nv = 2 + (seed as usize % 3); // 2..4 vars
        let num_polys = 1 + (seed as usize % 5); // 1..5 polys
        let num_points = 1 + (seed as usize % 3); // 1..3 distinct points

        let (pk, vk) = make_setup(1 << nv);

        let points: Vec<Vec<Fr>> = (0..num_points)
            .map(|_| (0..nv).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let polys: Vec<_> = (0..num_polys)
            .map(|_| Polynomial::<Fr>::random(nv, &mut rng))
            .collect();

        let claim_points: Vec<_> = (0..num_polys)
            .map(|i| points[i % num_points].clone())
            .collect();

        reduce_open_verify(&polys, &claim_points, &pk, &vk, b"kzg-property");
    }
}
