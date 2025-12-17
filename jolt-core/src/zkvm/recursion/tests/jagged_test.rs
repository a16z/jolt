//! Tests for the jagged polynomial bijection relation

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

    // Verify the evaluations match
    assert_eq!(sparse_sum, dense_eval, "Sparse and dense evaluations should match");
    assert_eq!(sparse_eval, sparse_sum, "MultilinearPolynomial evaluation should match manual sum");
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

    // Build the dense polynomial and get all components
    let (dense_poly, jagged_bijection, mapping) = constraint_system.build_dense_polynomial();

    // Test with random field element evaluation points
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;
    let num_dense_vars = dense_poly.get_num_vars();

    let zr: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zc: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Pre-compute all eq evaluations for efficiency
    let eq_row_evals = EqPolynomial::<Fq>::evals(&zr);
    let eq_col_evals = EqPolynomial::<Fq>::evals(&zc);

    // Compute sparse evaluation manually (since MultilinearPolynomial evaluation has different semantics)
    // The sparse matrix M(s, x) is evaluated as: sum_{s,x in {0,1}^n} M[s][x] * eq(s, zr) * eq(x, zc)
    let row_size = 1 << constraint_system.matrix.num_constraint_vars;
    let sparse_eval: Fq = (0..constraint_system.matrix.num_rows)
        .into_par_iter()
        .map(|s_idx| {
            let mut row_sum = Fq::zero();
            for x_idx in 0..row_size {
                let flat_idx = s_idx * row_size + x_idx;
                let val = constraint_system.matrix.evaluations[flat_idx];
                if !val.is_zero() {
                    let eq_s = eq_row_evals[s_idx];
                    let eq_x = eq_col_evals[x_idx];
                    row_sum += val * eq_s * eq_x;
                }
            }
            row_sum
        })
        .sum();

    // Compute dense evaluation using the bijection
    let dense_size =
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);

    let dense_eval: Fq = (0..dense_size)
        .into_par_iter()
        .map(|dense_idx| {
            // Get polynomial index and evaluation index from bijection
            let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, dense_idx);
            let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, dense_idx);

            // Decode to get constraint index and polynomial type
            let (constraint_idx, poly_type) = mapping.decode(poly_idx);

            // Get the matrix row index
            let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);

            // Get eq evaluations using the correct indices
            let eq_s = eq_row_evals[matrix_row];
            let eq_x = eq_col_evals[eval_idx];
            let q_val = dense_poly.Z[dense_idx];

            q_val * eq_s * eq_x
        })
        .sum();

    // Verify that sparse and dense evaluations match
    assert_eq!(sparse_eval, dense_eval, "Sparse and dense evaluations should match");
}
