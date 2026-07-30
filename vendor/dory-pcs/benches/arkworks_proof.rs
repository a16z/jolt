//! Arkworks backend proof generation benchmark
//!
//! This benchmark demonstrates the performance of Dory proof generation
//! with all arkworks backend optimizations enabled:
//! - Cache: Pre-computed prepared points for faster pairings (~20-30% speedup)
//! - Parallel: Parallelized multi-scalar multiplications using Rayon
//!
//! Benchmarks with 2^26 coefficients (nu=13, sigma=13, 26 variables)
//!
//! Run with: cargo bench --bench arkworks_proof --features backends,cache,parallel

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dory_pcs::backends::arkworks::{
    ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
};
use dory_pcs::mode::Transparent;
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::{prove, setup, verify};

#[cfg(feature = "cache")]
use dory_pcs::backends::arkworks::init_cache;

fn setup_benchmark_data() -> (
    ArkworksPolynomial,
    Vec<ArkFr>,
    dory_pcs::setup::ProverSetup<BN254>,
    dory_pcs::setup::VerifierSetup<BN254>,
) {
    let max_log_n = 26;

    let (prover_setup, verifier_setup) = setup::<BN254>(max_log_n);

    // Initialize cache with setup generators for optimized pairings
    #[cfg(feature = "cache")]
    {
        if !dory_pcs::backends::arkworks::is_cached() {
            init_cache(&prover_setup.g1_vec, &prover_setup.g2_vec);
        }
    }

    // Create polynomial with 2^26 coefficients (nu=13, sigma=13)
    let poly_size = 1 << 26; // 67,108,864 coefficients
    let num_vars = 26;
    let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
    let poly = ArkworksPolynomial::new(coefficients);

    let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();

    (poly, point, prover_setup, verifier_setup)
}

fn bench_commitment(c: &mut Criterion) {
    let (poly, _, prover_setup, _) = setup_benchmark_data();
    let nu = 13;
    let sigma = 13;

    c.bench_function("commitment_2^26_coefficients", |b| {
        b.iter(|| {
            poly.commit::<BN254, Transparent, G1Routines>(
                black_box(nu),
                black_box(sigma),
                black_box(&prover_setup),
            )
            .unwrap()
        })
    });
}

fn bench_prove(c: &mut Criterion) {
    let (poly, point, prover_setup, _) = setup_benchmark_data();
    let nu = 13;
    let sigma = 13;

    let (_, tier_1, commit_blind) = poly
        .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
        .unwrap();

    c.bench_function("prove_2^26_coefficients", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::new(b"dory-bench");
            prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
                black_box(&poly),
                black_box(&point),
                black_box(tier_1.clone()),
                black_box(commit_blind),
                black_box(nu),
                black_box(sigma),
                black_box(&prover_setup),
                black_box(&mut transcript),
            )
            .unwrap()
        })
    });
}

fn bench_verify(c: &mut Criterion) {
    let (poly, point, prover_setup, verifier_setup) = setup_benchmark_data();
    let nu = 13;
    let sigma = 13;

    let (tier_2, tier_1, commit_blind) = poly
        .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
        .unwrap();

    let mut prover_transcript = Blake2bTranscript::new(b"dory-bench");
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
        &poly,
        &point,
        tier_1,
        commit_blind,
        nu,
        sigma,
        &prover_setup,
        &mut prover_transcript,
    )
    .unwrap();

    let evaluation = poly.evaluate(&point);

    c.bench_function("verify_2^26_coefficients", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::new(b"dory-bench");
            verify::<_, BN254, G1Routines, G2Routines, _>(
                black_box(tier_2),
                black_box(evaluation),
                black_box(&point),
                black_box(&proof),
                black_box(verifier_setup.clone()),
                black_box(&mut transcript),
            )
            .unwrap()
        })
    });
}

fn bench_end_to_end(c: &mut Criterion) {
    let max_log_n = 26;
    let (prover_setup, verifier_setup) = setup::<BN254>(max_log_n);

    // Initialize cache once
    #[cfg(feature = "cache")]
    {
        if !dory_pcs::backends::arkworks::is_cached() {
            init_cache(&prover_setup.g1_vec, &prover_setup.g2_vec);
        }
    }

    c.bench_function("end_to_end_2^26_coefficients", |b| {
        b.iter(|| {
            let nu = 13;
            let sigma = 13;
            let poly_size = 1 << 26; // 67,108,864 coefficients
            let num_vars = 26;

            // Create polynomial
            let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
            let poly = ArkworksPolynomial::new(coefficients);

            // Commit
            let (tier_2, tier_1, commit_blind) = poly
                .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
                .unwrap();

            // Evaluate
            let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
            let evaluation = poly.evaluate(&point);

            // Prove
            let mut prover_transcript = Blake2bTranscript::new(b"dory-bench");
            let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
                &poly,
                &point,
                tier_1,
                commit_blind,
                nu,
                sigma,
                &prover_setup,
                &mut prover_transcript,
            )
            .unwrap();

            // Verify
            let mut verifier_transcript = Blake2bTranscript::new(b"dory-bench");
            verify::<_, BN254, G1Routines, G2Routines, _>(
                tier_2,
                evaluation,
                &point,
                &proof,
                verifier_setup.clone(),
                &mut verifier_transcript,
            )
            .unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_commitment,
    bench_prove,
    bench_verify,
    bench_end_to_end
);
criterion_main!(benches);
