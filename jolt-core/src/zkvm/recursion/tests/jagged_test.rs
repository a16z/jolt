//! Tests for the jagged polynomial bijection relation
//!
//! Tests equation (3) from the paper:
//! p̂(zr, zc) = Σ_{x∈{0,1}^n, y∈{0,1}^k} p(x, y) · eq(x, zr) · eq(y, zc)
//!          = Σ_{i∈{0,1}^m} q(i) · eq(rowt(i), zr) · eq(colt(i), zc)

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::recursion::{
        bijection::{JaggedPolynomial, JaggedTransform, VarCountJaggedBijection},
        RecursionProver,
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::{One, UniformRand, Zero};
use ark_std::test_rng;
use rayon::prelude::*;
use serial_test::serial;

/// Convert index to binary representation for polynomial evaluation
fn index_to_binary<F: JoltField>(idx: usize, num_vars: usize) -> Vec<F> {
    (0..num_vars)
        .map(|i| {
            if (idx >> i) & 1 == 1 {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect()
}

#[test]
fn test_jagged_relation_small() {
    // Create a small constraint system with mixed 2-var and 3-var polynomials
    // This allows us to exhaustively check the entire hypercube

    let polynomials = vec![
        JaggedPolynomial::new(2), // 4 evaluations
        JaggedPolynomial::new(3), // 8 evaluations
        JaggedPolynomial::new(2), // 4 evaluations
    ];

    let bijection = VarCountJaggedBijection::new(polynomials.clone());

    // Create sparse matrix p(x, y) where:
    // - x has 3 variables (to accommodate largest polynomial)
    // - y has 2 variables (log2(3 polynomials) rounded up)
    let sparse_rows = 3; // number of polynomials
    let sparse_cols = 8; // max polynomial size (2^3)
    let total_sparse_size = sparse_rows * sparse_cols;

    // Fill sparse matrix with test values
    let mut sparse_p = vec![Fq::zero(); total_sparse_size];
    let mut dense_q = Vec::new();

    // Fill each polynomial with distinct values
    for (poly_idx, poly) in polynomials.iter().enumerate() {
        for eval_idx in 0..poly.native_size {
            let sparse_idx = poly_idx * sparse_cols + eval_idx;
            let value = Fq::from((poly_idx * 100 + eval_idx + 1) as u64);
            sparse_p[sparse_idx] = value;

            // Add to dense polynomial (no padding)
            dense_q.push(value);
        }
        // Pad the rest of the sparse row with zeros
        for eval_idx in poly.native_size..sparse_cols {
            let sparse_idx = poly_idx * sparse_cols + eval_idx;
            sparse_p[sparse_idx] = Fq::zero();
        }
    }

    // Test with random evaluation points
    let mut rng = test_rng();

    // Create a MultilinearPolynomial from sparse evaluations
    // The sparse matrix has variables ordered as [col_vars, row_vars]
    // matching the constraint system's [x_vars, s_vars] ordering
    // Pad to power of 2
    let mut sparse_p_padded = sparse_p.clone();
    let padded_size = sparse_p_padded.len().next_power_of_two();
    sparse_p_padded.resize(padded_size, Fq::zero());
    let sparse_mlpoly = MultilinearPolynomial::from(sparse_p_padded);

    // Create evaluation point: [zc, zr] (col vars first, then row vars)
    // After padding to 32 elements, we need 5 total variables
    let num_total_vars = padded_size.trailing_zeros() as usize; // 5 for 32 elements
    let num_col_vars = 3; // log2(8) - columns still have 8 elements
    let num_row_vars = num_total_vars - num_col_vars; // This will be 2
    let zc: Vec<Fq> = (0..num_col_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zr: Vec<Fq> = (0..num_row_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Combine into single evaluation point [zc, zr]
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zc);
    eval_point.extend_from_slice(&zr);

    // Convert to challenges for polynomial evaluation
    // Note: MultilinearPolynomial expects challenges in reverse order (big-endian)
    let eval_challenges: Vec<<Fq as JoltField>::Challenge> = eval_point
        .iter()
        .rev()  // Reverse for correct endianness
        .map(|&x| x.into())
        .collect();

    // Evaluate sparse polynomial
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

    // Verify the jagged relation by checking that both representations give the same result
    // We'll sum over all boolean hypercube points to verify the relation

    // First, compute the sum using the sparse representation
    let mut sparse_sum = Fq::zero();
    for row_idx in 0..sparse_rows {
        for col_idx in 0..sparse_cols {
            let val = sparse_p[row_idx * sparse_cols + col_idx];
            if !val.is_zero() {
                // For this value, compute its contribution at the evaluation point
                let row_binary = index_to_binary::<Fq>(row_idx, num_row_vars);
                let col_binary = index_to_binary::<Fq>(col_idx, num_col_vars);
                let eq_row = EqPolynomial::mle(&row_binary, &zr);
                let eq_col = EqPolynomial::mle(&col_binary, &zc);
                sparse_sum += val * eq_row * eq_col;
            }
        }
    }

    // Now compute using the dense representation with the bijection
    let mut dense_sum = Fq::zero();
    for (i, &val) in dense_q.iter().enumerate() {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, i);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, i);

        let row_binary = index_to_binary::<Fq>(row, num_row_vars);
        let col_binary = index_to_binary::<Fq>(col, num_col_vars);
        let eq_row = EqPolynomial::mle(&row_binary, &zr);
        let eq_col = EqPolynomial::mle(&col_binary, &zc);

        dense_sum += val * eq_row * eq_col;
    }

    // These should be equal
    let dense_eval = dense_sum;

    // Verify that dense_q contains exactly the non-zero values from sparse_p
    let mut sparse_non_zero_values = Vec::new();
    for val in &sparse_p {
        if !val.is_zero() {
            sparse_non_zero_values.push(*val);
        }
    }
    assert_eq!(sparse_non_zero_values, dense_q,
        "Dense polynomial should contain exactly the non-zero values from sparse");

    // Now check the evaluations
    assert_eq!(
        sparse_sum, dense_eval,
        "Sparse sum {} != Dense sum {}",
        sparse_sum, dense_eval
    );

    // Verify that the multilinear polynomial evaluation matches our manual sum
    assert_eq!(
        sparse_eval, sparse_sum,
        "MultilinearPolynomial evaluation should match manual sum"
    );

    println!("✓ Small jagged relation test passed!");
    println!("  Sparse eval: {}", sparse_eval);
    println!("  Dense eval:  {}", dense_eval);
}

#[test]
#[serial]
fn test_jagged_relation_dory_witness() {
    // Initialize Dory for creating a real witness
    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let mut rng = test_rng();

    // Create a test Dory proof
    let num_vars = 4;
    let poly_coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly_coefficients));

    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);

    let (commitment, hint) =
        <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    let mut point_transcript: Blake2bTranscript = Transcript::new(b"test_point");
    let point_challenges: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();

    let evaluation = PolynomialEvaluation::evaluate(&poly, &point_challenges);

    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");
    let opening_proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &point_challenges,
        Some(hint),
        &mut prover_transcript,
    );

    // Create RecursionProver from Dory proof
    let gamma = Fq::rand(&mut rng);
    let delta = Fq::rand(&mut rng);

    let mut witness_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");

    use crate::poly::commitment::dory::wrappers::{ArkDoryProof, ArkGT};

    let ark_proof = ArkDoryProof::from(opening_proof);
    let ark_commitment = ArkGT::from(commitment);

    let prover = RecursionProver::<Fq>::new_from_dory_proof(
        &ark_proof,
        &verifier_setup,
        &mut witness_transcript,
        &point_challenges,
        &evaluation,
        &ark_commitment,
        gamma,
        delta,
    )
    .expect("Failed to create recursion prover");

    // Get the constraint system and build dense polynomial
    let constraint_system = &prover.constraint_system;
    let (dense_poly, jagged_bijection) = constraint_system.build_dense_polynomial();

    // Test with random field element evaluation points
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;
    let num_dense_vars = dense_poly.get_num_vars();

    let zr: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zc: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Debug: Print matrix structure
    println!("Matrix structure:");
    println!("  - num_s_vars: {}", num_s_vars);
    println!("  - num_x_vars: {}", num_x_vars);
    println!(
        "  - num_constraints: {}",
        constraint_system.num_constraints()
    );
    println!(
        "  - num_constraints_padded: {}",
        constraint_system.matrix.num_constraints_padded
    );
    println!(
        "  - matrix evaluations len: {}",
        constraint_system.matrix.evaluations.len()
    );
    println!(
        "  - dense size: {}",
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection)
    );

    // Compute sparse evaluation using parallel iterators
    let num_constraints = constraint_system.num_constraints();
    let num_constraints_padded = constraint_system.matrix.num_constraints_padded;
    let num_evals_per_constraint = 1 << num_x_vars;

    // Debug: Let's understand the matrix layout
    println!("\nMatrix layout debug:");
    println!("  - Total matrix evaluations: {}", constraint_system.matrix.evaluations.len());
    println!("  - Expected size: {} * {} = {}",
        constraint_system.matrix.num_constraints_padded,
        1 << num_x_vars,
        constraint_system.matrix.num_constraints_padded * (1 << num_x_vars)
    );

    // The matrix should have num_constraints_padded rows and 2^num_x_vars columns
    // Total variables = log2(matrix size) = log2(33554432) = 25
    let matrix_total_vars = constraint_system.matrix.evaluations.len().trailing_zeros() as usize;
    println!("  - Matrix total vars: {}", matrix_total_vars);
    println!("  - Matrix has {} s-vars and {} x-vars", num_s_vars, num_x_vars);

    // Create a MultilinearPolynomial from the constraint system's matrix evaluations
    // With zero padding, the matrix directly follows the paper's semantics
    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());

    // Create evaluation point: [zc, zr] (col vars first, then row vars)
    // Based on the matrix structure, variables should be ordered as [x_vars, s_vars]
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zc);
    eval_point.extend_from_slice(&zr);

    // Convert to challenges for polynomial evaluation
    // Note: MultilinearPolynomial expects challenges in reverse order (big-endian)
    let eval_challenges: Vec<<Fq as JoltField>::Challenge> = eval_point
        .iter()
        .rev()  // Reverse for correct endianness
        .map(|&x| x.into())
        .collect();

    // Evaluate sparse polynomial
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

    println!("  - Sparse polynomial evaluation: {}", sparse_eval);

    // Debug: Let's check how padding works in the sparse matrix
    println!("\nDebug: Checking padding in sparse matrix:");

    // Get the first GT exp polynomial (4-var, should be padded to 8-var)
    let first_gt_exp_row = 0; // First row in matrix
    let storage_offset = constraint_system.matrix.storage_offset(first_gt_exp_row);

    // Check the padding pattern: pad_4var_to_8var repeats EACH value 16 times consecutively
    println!("  First GT exp polynomial values (checking consecutive repetition):");

    // For pad_4var_to_8var: mle_4var[i] is repeated 16 times at positions [i*16..(i+1)*16]
    let mut has_correct_padding = true;
    for mle_idx in 0..16 {
        let start_pos = mle_idx * 16;
        let base_val = constraint_system.matrix.evaluations[storage_offset + start_pos];

        if mle_idx < 2 {
            println!("    MLE[{}] value = {} (at positions {}-{})",
                mle_idx, base_val, start_pos, start_pos + 15);
        }

        // Check that this value is repeated 16 times
        for offset in 1..16 {
            let pos = start_pos + offset;
            let val = constraint_system.matrix.evaluations[storage_offset + pos];
            if val != base_val {
                has_correct_padding = false;
                println!("    WARNING: Position {} has value {}, expected {}",
                    pos, val, base_val);
            }
        }
    }
    println!("  Has correct padding pattern: {}", has_correct_padding);

    // Debug: Let's check the bijection more carefully
    println!("\nDebug: Understanding the bijection:");

    // Check how many polynomials we have and their sizes
    println!("  Total polynomials in bijection: {}", jagged_bijection.num_polynomials());

    // Check the first few polynomial sizes
    for poly_idx in 0..5.min(jagged_bijection.num_polynomials()) {
        let num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(&jagged_bijection, poly_idx);
        let poly_size = 1 << num_vars;
        println!("  Polynomial[{}]: {} vars, size {}", poly_idx, num_vars, poly_size);
    }

    // Now let's see what the dense polynomial looks like
    let dense_size_check = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);
    println!("\nDebug: Dense polynomial extraction:");
    println!("  Dense size: {}", dense_size_check);

    // Check the first polynomial's extraction
    println!("  First polynomial extraction (should extract 16 unique values for 4-var):");
    for i in 0..20.min(dense_size_check) {
        let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
        let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);
        let dense_val = dense_poly.Z[i];

        if i < 5 || i == 15 || i == 16 {
            println!("    Dense[{}] -> poly[{}], eval[{}] = {}",
                i, row, col, dense_val);
        }
    }

    // Pre-compute all eq evaluations for efficiency
    let eq_row_evals = EqPolynomial::<Fq>::evals(&zr);
    let eq_col_evals = EqPolynomial::<Fq>::evals(&zc);

    // Compute dense evaluation using the bijection (also in parallel)
    let dense_size =
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);

    let dense_eval: Fq = (0..dense_size)
        .into_par_iter()
        .map(|i| {
            // Get row and col using bijection
            let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
            let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);

            // Use pre-computed eq evaluations
            let eq_row = eq_row_evals[row];
            let eq_col = eq_col_evals[col];

            let q_val = dense_poly.Z[i];
            q_val * eq_row * eq_col
        })
        .sum();

    // The evaluations should be equal
    assert_eq!(
        sparse_eval, dense_eval,
        "Sparse eval {} != Dense eval {}",
        sparse_eval, dense_eval
    );

    println!("✓ Dory witness jagged relation test passed!");
    println!("  Sparse eval: {}", sparse_eval);
    println!("  Dense eval:  {}", dense_eval);
    println!(
        "  Dense size: {} (from {} sparse entries)",
        dense_size,
        constraint_system.matrix.evaluations.len()
    );
}
