#![allow(missing_docs)]
use ark_bn254::Fr;
use ark_ff::UniformRand;
use dory::*;
use std::time::Instant;

use dory::curve::{
    test_rng, ArkBn254Pairing, DummyMsm, OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial,
};

#[test]
fn test_pcs_api_workflow() {
    let mut rng = test_rng();
    let domain = b"dory_pcs_test";

    // Multilinear polynomial parameters
    let num_variables = 24;
    let sigma = 12; // sigma must be <= max_log_n / 2 for the SRS
    let num_coeffs = 1 << num_variables;

    println!(
        "Testing PCS API with {} variables, {} coefficients, sigma = {}",
        num_variables, num_coeffs, sigma
    );

    // Setup with preloaded SRS file
    let setup_start = Instant::now();
    let srs_path = "./k_12.srs";
    let (prover_setup, verifier_setup) =
        setup_with_srs_file::<ArkBn254Pairing, _>(&mut rng, num_variables, Some(srs_path));

    // Initialize cache for performance optimization
    println!("Initializing cache...");
    let cache_init_start = Instant::now();
    // prover_setup.init_cache();
    let cache_init_time = cache_init_start.elapsed();
    println!("✓ Cache initialized in {:?}", cache_init_time);
    println!(
        "Cache initialization complete. Has cache: {}",
        prover_setup.has_cache()
    );

    // Print memory usage statistics for the caches
    println!("\n=== CACHE MEMORY PROFILING ===");
    if let Some(g1_cache) = &prover_setup.g1_cache {
        g1_cache.print_memory_stats();
    }
    if let Some(g2_cache) = &prover_setup.g2_cache {
        g2_cache.print_memory_stats();
    }

    let setup_time = setup_start.elapsed();
    println!("Setup time (including cache): {:?}", setup_time);

    // Random multilinear polynomial coefficients
    let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

    // Random evaluation point (one value per variable)
    let point: Vec<Fr> = (0..num_variables).map(|_| Fr::rand(&mut rng)).collect();

    // Commit to polynomial
    let commit_start = Instant::now();
    let polynomial = StandardPolynomial::new(&coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);
    let commit_time = commit_start.elapsed();
    println!("Commit time: {:?}", commit_time);

    // Evaluate and prove
    let eval_start = Instant::now();
    let transcript = create_transcript(domain);
    let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    let evaluation = polynomial.evaluate(&point);
    let eval_time = eval_start.elapsed();
    println!("Evaluate and prove time: {:?}", eval_time);

    // Print proof statistics before verification consumes it
    proof.print_proof_stats();

    // Verify - create fresh transcript for verification
    let verify_start = Instant::now();
    let verify_transcript = create_transcript(domain);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        verify_transcript,
    );
    let verify_time = verify_start.elapsed();
    println!("Verify time: {:?}", verify_time);

    let total_time = setup_time + commit_time + eval_time + verify_time;
    println!("Total time: {:?}", total_time);

    assert!(result.is_ok(), "PCS verification should succeed");
    println!("✓ PCS API test passed");
}
