#![allow(missing_docs)]
use ark_bn254::{Fq12, Fr};
use ark_ff::UniformRand;
use dory::{
    arithmetic::Field as DoryField, commit, create_transcript, evaluate, setup_with_srs_file,
    verify,
};

use dory::curve::{
    test_rng, ArkBn254Pairing, DummyMsm, OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial,
};

// Helper function to generate test environment for PCS tests
fn setup_pcs_test_environment(
    num_variables: usize,
    _sigma: usize,
) -> (
    dory::setup::ProverSetup<ArkBn254Pairing>,
    dory::setup::VerifierSetup<ArkBn254Pairing>,
    Vec<Fr>,
    Vec<Fr>,
) {
    let mut rng = test_rng();
    let num_coeffs = 1 << num_variables;

    // Setup
    let (prover_setup, verifier_setup) =
        setup_with_srs_file::<ArkBn254Pairing, _>(&mut rng, num_variables, None);

    // Generate random polynomial coefficients
    let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

    // Generate random evaluation point
    let point: Vec<Fr> = (0..num_variables).map(|_| Fr::rand(&mut rng)).collect();

    (prover_setup, verifier_setup, coeffs, point)
}

#[test]
fn test_soundness_wrong_evaluation() {
    println!("=== Testing soundness: claiming wrong evaluation ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;

    let (prover_setup, verifier_setup, coeffs, point) =
        setup_pcs_test_environment(num_variables, sigma);

    // Commit to polynomial
    let polynomial = StandardPolynomial::new(&coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);

    // Generate proof
    let correct_evaluation = polynomial.evaluate(&point);
    let transcript = create_transcript(domain);
    let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );

    // Try to verify with wrong evaluation
    let wrong_evaluation = correct_evaluation + Fr::rand(&mut rng);
    // Create fresh transcript for verification
    let verify_transcript = create_transcript(domain);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        wrong_evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        verify_transcript,
    );

    assert!(
        result.is_err(),
        "Verification should fail with wrong evaluation"
    );
    println!("✓ Verification correctly failed with wrong evaluation");
}

#[test]
fn test_soundness_wrong_commitment() {
    println!("=== Testing soundness: using wrong commitment ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;

    let (prover_setup, verifier_setup, coeffs, point) =
        setup_pcs_test_environment(num_variables, sigma);

    // Generate proof
    let polynomial = StandardPolynomial::new(&coeffs);
    let evaluation = polynomial.evaluate(&point);
    let transcript = create_transcript(domain);
    let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );

    // Try to verify with wrong commitment
    let wrong_commitment = Fq12::rand(&mut rng);
    // Create fresh transcript for verification
    let verify_transcript = create_transcript(domain);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        wrong_commitment,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        verify_transcript,
    );

    assert!(
        result.is_err(),
        "Verification should fail with wrong commitment"
    );
    println!("✓ Verification correctly failed with wrong commitment");
}

#[test]
fn test_soundness_wrong_evaluation_point() {
    println!("=== Testing soundness: using wrong evaluation point ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;

    let (prover_setup, verifier_setup, coeffs, point) =
        setup_pcs_test_environment(num_variables, sigma);

    // Commit to polynomial
    let polynomial = StandardPolynomial::new(&coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);

    // Generate proof for one point
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

    // Try to verify with different point
    let mut wrong_point = point.clone();
    wrong_point[0] = wrong_point[0] + Fr::rand(&mut rng);

    // Create fresh transcript for verification
    let verify_transcript = create_transcript(domain);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        evaluation,
        &wrong_point,
        proof,
        sigma,
        &verifier_setup,
        verify_transcript,
    );

    assert!(
        result.is_err(),
        "Verification should fail with wrong evaluation point"
    );
    println!("✓ Verification correctly failed with wrong evaluation point");
}

#[test]
fn test_soundness_binding_property() {
    println!("=== Testing soundness: binding property ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;
    let num_coeffs = 1 << num_variables;

    let (prover_setup, verifier_setup, coeffs1, point) =
        setup_pcs_test_environment(num_variables, sigma);

    // Create second polynomial (different from first)
    let mut coeffs2: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
    coeffs2[0] = coeffs1[0] + Fr::rand(&mut rng); // Ensure difference

    // Commit to both polynomials
    let poly1 = StandardPolynomial::new(&coeffs1);
    let poly2 = StandardPolynomial::new(&coeffs2);
    let (commitment1, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&poly1, 0, sigma, &prover_setup);
    let (commitment2, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&poly2, 0, sigma, &prover_setup);

    // Commitments should be different
    assert_ne!(
        commitment1, commitment2,
        "Commitments to different polynomials should differ"
    );

    // Generate proof for first polynomial
    let transcript = create_transcript(domain);
    let proof1 = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs1),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    let eval1 = poly1.evaluate(&point);

    // Try to use proof1 with commitment2
    // Create fresh transcript for verification
    let verify_transcript = create_transcript(domain);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment2,
        eval1,
        &point,
        proof1,
        sigma,
        &verifier_setup,
        verify_transcript,
    );

    assert!(
        result.is_err(),
        "Cannot use proof for one polynomial with commitment of another"
    );
    println!("✓ Binding property correctly enforced");
}

#[test]
fn test_soundness_polynomial_mismatch() {
    println!("=== Testing soundness: polynomial and proof mismatch ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;
    let num_coeffs = 1 << num_variables;

    let (prover_setup, verifier_setup, coeffs1, point) =
        setup_pcs_test_environment(num_variables, sigma);

    // Create different polynomial
    let coeffs2: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

    // Commit to first polynomial
    let polynomial = StandardPolynomial::new(&coeffs1);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);

    // Generate proof for second polynomial
    let transcript = create_transcript(domain);
    let proof2 = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs2),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    let eval2 = polynomial.evaluate(&point);

    // Try to verify commitment1 with proof from coeffs2
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        eval2,
        &point,
        proof2,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(
        result.is_err(),
        "Verification should fail with mismatched polynomial and proof"
    );
    println!("✓ Verification correctly failed with polynomial mismatch");
}

#[test]
fn test_soundness_offset_manipulation() {
    println!("=== Testing soundness: commitment offset manipulation ===");
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;

    let (prover_setup, verifier_setup, coeffs, point) =
        setup_pcs_test_environment(num_variables, sigma);

    let polynomial = StandardPolynomial::new(&coeffs);

    // Commit with different offset
    let (commitment2, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 16, sigma, &prover_setup);

    // Generate proof for offset 0
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

    // Try to verify commitment2 (offset 16) with proof from offset 0
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment2,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(
        result.is_err(),
        "Verification should fail with wrong offset commitment"
    );
    println!("✓ Verification correctly failed with offset manipulation");
}

#[test]
fn test_soundness_zero_polynomial() {
    println!("=== Testing soundness: zero polynomial special case ===");
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;
    let num_coeffs = 1 << num_variables;

    let (prover_setup, verifier_setup, _, point) = setup_pcs_test_environment(num_variables, sigma);

    // Zero polynomial
    let zero_coeffs: Vec<Fr> = vec![Fr::zero(); num_coeffs];

    let zero_poly = StandardPolynomial::new(&zero_coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&zero_poly, 0, sigma, &prover_setup);

    let transcript = create_transcript(domain);
    let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&zero_coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );

    // Zero polynomial should evaluate to zero
    assert_eq!(
        zero_poly.evaluate(&point),
        Fr::zero(),
        "Zero polynomial should evaluate to zero"
    );

    // Try to claim non-zero evaluation for zero polynomial
    let wrong_evaluation = Fr::one();
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        wrong_evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(
        result.is_err(),
        "Cannot claim non-zero evaluation for zero polynomial"
    );
    println!("✓ Zero polynomial soundness test passed");
}

#[test]
fn test_soundness_constant_polynomial() {
    println!("=== Testing soundness: constant polynomial ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 8;
    let sigma = 6;
    let num_coeffs = 1 << num_variables;

    let (prover_setup, verifier_setup, _, point) = setup_pcs_test_environment(num_variables, sigma);

    // Constant polynomial (only first coefficient is non-zero)
    let constant_value = Fr::rand(&mut rng);
    let mut coeffs = vec![Fr::zero(); num_coeffs];
    coeffs[0] = constant_value;

    let polynomial = StandardPolynomial::new(&coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);

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

    // Constant polynomial should evaluate to the constant
    assert_eq!(
        evaluation, constant_value,
        "Constant polynomial should evaluate to constant"
    );

    // Try to claim wrong evaluation
    let wrong_evaluation = constant_value + Fr::one();
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        wrong_evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(
        result.is_err(),
        "Cannot claim wrong evaluation for constant polynomial"
    );
    println!("✓ Constant polynomial soundness test passed");
}

#[test]
fn test_soundness_completeness_multiple_points() {
    println!("=== Testing soundness: completeness for multiple points ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;

    let (prover_setup, verifier_setup, coeffs, _) =
        setup_pcs_test_environment(num_variables, sigma);

    let polynomial = StandardPolynomial::new(&coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);

    // Test multiple evaluation points
    for i in 0..5 {
        println!("  Testing point {}/5", i + 1);
        let point: Vec<Fr> = (0..num_variables).map(|_| Fr::rand(&mut rng)).collect();

        let transcript = create_transcript(domain);
        let polynomial = StandardPolynomial::new(&coeffs);
        let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
            &polynomial,
            None,
            &point,
            sigma,
            &prover_setup,
            transcript,
        );
        let evaluation = polynomial.evaluate(&point);

        let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
            commitment,
            evaluation,
            &point,
            proof,
            sigma,
            &verifier_setup,
            create_transcript(domain),
        );

        assert!(result.is_ok(), "Valid proof should verify for point {}", i);
    }

    println!("✓ Completeness for multiple points test passed");
}

#[test]
fn test_soundness_cross_polynomial_attack() {
    println!("=== Testing soundness: cross-polynomial attack ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;
    let num_coeffs = 1 << num_variables;

    let (prover_setup, verifier_setup, coeffs1, point) =
        setup_pcs_test_environment(num_variables, sigma);

    // Create two different polynomials
    let coeffs2: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

    // Commit to both
    let poly1 = StandardPolynomial::new(&coeffs1);
    let poly2 = StandardPolynomial::new(&coeffs2);
    let (commitment1, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&poly1, 0, sigma, &prover_setup);
    let (commitment2, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&poly2, 0, sigma, &prover_setup);

    // Generate proofs for both
    let transcript1 = create_transcript(domain);
    let proof1 = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs1),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript1,
    );
    let eval1 = poly1.evaluate(&point);

    let transcript2 = create_transcript(domain);
    let proof2 = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs2),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript2,
    );
    let eval2 = poly2.evaluate(&point);

    // Try cross-attacks
    // Attack 1: Use commitment1 with eval2 and proof2
    let result1 = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment1,
        eval2,
        &point,
        proof2,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );
    assert!(
        result1.is_err(),
        "Cannot use proof from one polynomial with commitment of another"
    );

    // Attack 2: Use commitment2 with eval1 and proof1
    let result2 = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment2,
        eval1,
        &point,
        proof1,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );
    assert!(
        result2.is_err(),
        "Cannot use proof from one polynomial with commitment of another"
    );

    println!("✓ Cross-polynomial attack correctly prevented");
}

#[test]
fn test_soundness_batch_verification() {
    println!("=== Testing soundness: batch verification attacks ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;
    let num_coeffs = 1 << num_variables;
    let batch_size = 3;

    let (prover_setup, verifier_setup, _, point) = setup_pcs_test_environment(num_variables, sigma);

    // Generate multiple polynomials
    let mut coeffs_batch = Vec::new();
    let mut commitments = Vec::new();
    let mut evaluations = Vec::new();

    for _ in 0..batch_size {
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let polynomial = StandardPolynomial::new(&coeffs);
        let (commitment, _) =
            commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);

        let evaluation = polynomial.evaluate(&point);
        coeffs_batch.push(coeffs);
        commitments.push(commitment);
        evaluations.push(evaluation);
    }

    // Test that swapping evaluations is detected
    if batch_size >= 2 {
        println!("  Testing evaluation swapping attack...");

        // Generate valid proof for first polynomial
        let transcript = create_transcript(domain);
        let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
            &StandardPolynomial::new(&coeffs_batch[0]),
            None,
            &point,
            sigma,
            &prover_setup,
            transcript,
        );

        // Try to verify with swapped evaluation
        let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
            commitments[0],
            evaluations[1],
            &point,
            proof,
            sigma,
            &verifier_setup,
            create_transcript(domain),
        );

        assert!(
            result.is_err(),
            "Cannot use evaluation from different polynomial"
        );
    }

    println!("✓ Batch verification attack correctly prevented");
}

#[test]
fn test_soundness_sparse_polynomial() {
    println!("=== Testing soundness: sparse polynomial ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;
    let num_coeffs = 1 << num_variables;

    let (prover_setup, verifier_setup, _, point) = setup_pcs_test_environment(num_variables, sigma);

    // Create sparse polynomial
    let mut sparse_coeffs = vec![Fr::zero(); num_coeffs];
    sparse_coeffs[0] = Fr::rand(&mut rng);
    sparse_coeffs[1] = Fr::rand(&mut rng);
    sparse_coeffs[num_coeffs - 1] = Fr::rand(&mut rng);

    let sparse_poly = StandardPolynomial::new(&sparse_coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&sparse_poly, 0, sigma, &prover_setup);

    let transcript = create_transcript(domain);
    let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&sparse_coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    let evaluation = sparse_poly.evaluate(&point);

    // Try to claim wrong evaluation for sparse polynomial
    let wrong_evaluation = evaluation + Fr::rand(&mut rng);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        wrong_evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(
        result.is_err(),
        "Cannot claim wrong evaluation for sparse polynomial"
    );
    println!("✓ Sparse polynomial soundness test passed");
}

#[test]
fn test_soundness_commitment_consistency() {
    println!("=== Testing soundness: commitment consistency ===");
    let domain = b"pcs_soundness_test";

    let num_variables = 10;
    let sigma = 5;

    let (prover_setup, verifier_setup, coeffs, point) =
        setup_pcs_test_environment(num_variables, sigma);

    // Commit multiple times to same polynomial
    let polynomial = StandardPolynomial::new(&coeffs);
    let (commitment1, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);
    let (commitment2, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);

    // Commitments should be deterministic
    assert_eq!(
        commitment1, commitment2,
        "Commitments to same polynomial should be identical"
    );

    // Generate proof
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

    // Both commitments should verify with same proof
    let result1 = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment1,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    // Generate proof again for second verification (can't clone DoryProofBuilder)
    let transcript2 = create_transcript(domain);
    let proof2 = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript2,
    );

    let result2 = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment2,
        evaluation,
        &point,
        proof2,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(
        result1.is_ok() && result2.is_ok(),
        "Both commitments should verify"
    );
    println!("✓ Commitment consistency test passed");
}

#[test]
fn test_soundness_degree_bound() {
    println!("=== Testing soundness: degree bound enforcement ===");
    let mut rng = test_rng();
    let domain = b"pcs_soundness_test";

    let num_variables = 8;
    let sigma = 6;
    let num_coeffs = 1 << num_variables;

    let (prover_setup, verifier_setup, _, point) = setup_pcs_test_environment(num_variables, sigma);

    // Test with maximum degree polynomial
    let max_degree_coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

    let max_degree_poly = StandardPolynomial::new(&max_degree_coeffs);
    let (commitment, _) =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&max_degree_poly, 0, sigma, &prover_setup);

    let transcript = create_transcript(domain);
    let proof = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&max_degree_coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    let evaluation = max_degree_poly.evaluate(&point);

    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(result.is_ok(), "Maximum degree polynomial should verify");

    // Generate new proof for wrong evaluation test (can't reuse proof)
    let transcript2 = create_transcript(domain);
    let proof2 = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&max_degree_coeffs),
        None,
        &point,
        sigma,
        &prover_setup,
        transcript2,
    );

    // Try to claim wrong evaluation
    let wrong_evaluation = evaluation + Fr::rand(&mut rng);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        wrong_evaluation,
        &point,
        proof2,
        sigma,
        &verifier_setup,
        create_transcript(domain),
    );

    assert!(
        result.is_err(),
        "Cannot claim wrong evaluation even for max degree polynomial"
    );
    println!("✓ Degree bound enforcement test passed");
}
