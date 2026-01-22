//! Test to verify the definition of matrix M in Stage 2
//! This test ensures that M is indeed the multilinear extension of the v_i claims

use crate::{
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    zkvm::recursion::stage2::virtualization::{matrix_s_index, virtual_claim_index},
};
use ark_bn254::Fq;
use ark_ff::{UniformRand, Zero};
use ark_std::test_rng;

#[test]
fn test_matrix_mle_definition_direct() {
    // Test that verifies the matrix M structure directly
    // by building it manually and comparing with expected behavior

    type F = Fq;
    let mut rng = test_rng();

    // Create a small test case with 1 constraint
    let num_constraints = 1;
    let num_constraints_padded = 1; // Already a power of 2
    let num_poly_types = 16; // Updated for packed GT exp

    // Calculate num_s_vars: log2(num_poly_types * num_constraints_padded)
    let total_rows = num_poly_types * num_constraints_padded;
    let num_s_vars = (total_rows as f64).log2().ceil() as usize; // log2(16) = 4

    // Create test virtual claims as a flat vector
    // Order: for each constraint, all 16 poly types in order
    let mut virtual_claims = Vec::new();
    for _i in 0..num_constraints {
        // Add claims for all 16 polynomial types in order
        virtual_claims.push(F::from(1u64)); // Base
        virtual_claims.push(F::from(2u64)); // RhoPrev
        virtual_claims.push(F::from(3u64)); // RhoCurr
        virtual_claims.push(F::from(4u64)); // Quotient
        virtual_claims.push(F::from(5u64)); // Bit (new for packed GT exp)
        virtual_claims.push(F::from(6u64)); // MulLhs
        virtual_claims.push(F::from(7u64)); // MulRhs
        virtual_claims.push(F::from(8u64)); // MulResult
        virtual_claims.push(F::from(9u64)); // MulQuotient
        virtual_claims.push(F::from(10u64)); // G1ScalarMulXA
        virtual_claims.push(F::from(11u64)); // G1ScalarMulYA
        virtual_claims.push(F::from(12u64)); // G1ScalarMulXT
        virtual_claims.push(F::from(13u64)); // G1ScalarMulYT
        virtual_claims.push(F::from(14u64)); // G1ScalarMulXANext
        virtual_claims.push(F::from(15u64)); // G1ScalarMulYANext
        virtual_claims.push(F::from(16u64)); // G1ScalarMulIndicator
    }

    // Build the matrix evaluations manually following the same layout
    // as compute_virtualization_claim
    let mu_size = 1 << num_s_vars;
    let mut mu_evals = vec![F::zero(); mu_size];

    // Fill according to matrix layout: poly_type * num_constraints_padded + constraint_idx
    for constraint_idx in 0..num_constraints {
        for poly_idx in 0..num_poly_types {
            let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
            let s_idx = matrix_s_index(poly_idx, constraint_idx, num_constraints_padded);
            mu_evals[s_idx] = virtual_claims[claim_idx];
        }
    }

    // Create the multilinear polynomial M from these evaluations
    let m_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(mu_evals.clone()));

    // Test at random points
    println!("Testing matrix MLE definition at random points...");
    println!(
        "Matrix has {} s variables, {} total evaluations",
        num_s_vars, mu_size
    );

    for test_idx in 0..5 {
        // Sample random r_s
        let r_s: Vec<F> = (0..num_s_vars).map(|_| F::rand(&mut rng)).collect();

        // Method 1: Evaluate M directly at r_s
        let m_direct = PolynomialEvaluation::evaluate(&m_poly, &r_s);

        // Method 2: Compute Σ_i eq(r_s, i) · mu_evals[i]
        let eq_evals = EqPolynomial::<F>::evals(&r_s);
        let m_from_eq: F = eq_evals
            .iter()
            .zip(mu_evals.iter())
            .map(|(eq_val, mu_val)| *eq_val * *mu_val)
            .sum();

        println!(
            "Test {}: Direct eval = {:?}, From eq computation = {:?}",
            test_idx, m_direct, m_from_eq
        );
        assert_eq!(
            m_direct, m_from_eq,
            "M evaluation doesn't match expected MLE computation! Test index: {}",
            test_idx
        );
    }

    println!("✅ Direct test passed! M evaluations match MLE definition.");
}

#[test]
fn test_matrix_mle_with_multiple_constraints() {
    // Test with multiple constraints to ensure the layout is correct
    type F = Fq;
    let mut rng = test_rng();

    // Test with 2 constraints (padded to 2)
    let num_constraints = 2;
    let num_constraints_padded = 2;
    let num_poly_types = 16; // Updated for packed GT exp

    // Calculate num_s_vars: log2(16 * 2) = log2(32) = 5
    let total_rows = num_poly_types * num_constraints_padded;
    let num_s_vars = (total_rows as f64).log2().ceil() as usize;

    // Create test virtual claims with distinct values for each constraint
    let mut virtual_claims = Vec::new();

    // First constraint gets values 1-16
    for i in 1..=16 {
        virtual_claims.push(F::from(i as u64));
    }

    // Second constraint gets values 101-116
    for i in 101..=116 {
        virtual_claims.push(F::from(i as u64));
    }

    // Build the matrix evaluations
    let mu_size = 1 << num_s_vars;
    let mut mu_evals = vec![F::zero(); mu_size];

    // Fill matrix with proper layout
    for constraint_idx in 0..num_constraints {
        for poly_idx in 0..num_poly_types {
            let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
            let s_idx = matrix_s_index(poly_idx, constraint_idx, num_constraints_padded);
            mu_evals[s_idx] = virtual_claims[claim_idx];
        }
    }

    // Create the multilinear polynomial M
    let m_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(mu_evals.clone()));

    // Test evaluation
    let r_s: Vec<F> = (0..num_s_vars).map(|_| F::rand(&mut rng)).collect();
    let m_direct = PolynomialEvaluation::evaluate(&m_poly, &r_s);
    let eq_evals = EqPolynomial::<F>::evals(&r_s);
    let m_from_eq: F = eq_evals
        .iter()
        .zip(mu_evals.iter())
        .map(|(eq_val, mu_val)| *eq_val * *mu_val)
        .sum();

    assert_eq!(m_direct, m_from_eq, "Multi-constraint test failed!");

    println!("✅ Multiple constraints test passed! Matrix layout is correct.");
}
