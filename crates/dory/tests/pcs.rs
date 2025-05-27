#![allow(missing_docs)]
use ark_bn254::Fr;
use ark_ff::UniformRand;
use dory::*;
use std::time::Instant;

use dory::curve::{test_rng, ArkBn254Pairing, DummyMsm, OptimizedMsmG1, OptimizedMsmG2};

#[test]
fn test_pcs_api_workflow() {
    let mut rng = test_rng();
    let domain = b"dory_pcs_test";

    // Multilinear polynomial parameters
    let num_variables = 10;
    let sigma = 5;
    let num_coeffs = 1 << num_variables;

    println!(
        "Testing PCS API with {} variables, {} coefficients, sigma = {}",
        num_variables, num_coeffs, sigma
    );

    // Setup with preloaded SRS file
    let setup_start = Instant::now();
    let srs_path = "./k_10.srs";
    let (prover_setup, verifier_setup) =
        setup_with_srs_file::<ArkBn254Pairing, _>(&mut rng, num_variables, Some(srs_path));
    let setup_time = setup_start.elapsed();
    println!("Setup time: {:?}", setup_time);

    // Random multilinear polynomial coefficients
    let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

    // Random evaluation point (one value per variable)
    let point: Vec<Fr> = (0..num_variables).map(|_| Fr::rand(&mut rng)).collect();

    // Commit to polynomial
    let commit_start = Instant::now();
    let commitment = commit::<ArkBn254Pairing, OptimizedMsmG1>(&coeffs, 0, sigma, &prover_setup);
    let commit_time = commit_start.elapsed();
    println!("Commit time: {:?}", commit_time);

    // Evaluate and prove
    let eval_start = Instant::now();
    let transcript = create_transcript::<Fr>(domain);
    let (evaluation, proof) = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2>(
        &coeffs,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    let eval_time = eval_start.elapsed();
    println!("Evaluate and prove time: {:?}", eval_time);

    // Print proof statistics before verification consumes it
    proof.print_proof_stats();

    // Verify
    let verify_start = Instant::now();
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        domain,
    );
    let verify_time = verify_start.elapsed();
    println!("Verify time: {:?}", verify_time);

    let total_time = setup_time + commit_time + eval_time + verify_time;
    println!("Total time: {:?}", total_time);

    assert!(result.is_ok(), "PCS verification should succeed");
    println!("✓ PCS API test passed");
}
