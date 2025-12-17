//! Debug test to understand the jagged relation issues

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
        bijection::{JaggedTransform, VarCountJaggedBijection},
        RecursionProver,
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::{One, UniformRand, Zero};
use ark_std::test_rng;

#[test]
fn debug_jagged_relation() {
    // Initialize Dory
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

    let constraint_system = &prover.constraint_system;
    let (dense_poly, jagged_bijection, _mapping) = constraint_system.build_dense_polynomial();

    // Debug: Print matrix dimensions
    println!("=== Matrix Dimensions ===");
    println!("num_s_vars: {}", constraint_system.num_s_vars());
    println!("num_x_vars: {}", constraint_system.matrix.num_constraint_vars);
    println!("num_constraints: {}", constraint_system.matrix.num_constraints);
    println!("num_constraints_padded: {}", constraint_system.matrix.num_constraints_padded);
    println!("num_rows: {}", constraint_system.matrix.num_rows);
    println!("total matrix size: {}", constraint_system.matrix.evaluations.len());
    println!("dense poly size: {}", dense_poly.Z.len());
    println!("dense poly num_vars: {}", dense_poly.get_num_vars());

    // Debug: Check the sparse matrix structure
    println!("\n=== Checking Sparse Matrix Structure ===");
    let row_size = 1 << constraint_system.matrix.num_constraint_vars;
    println!("row_size: {}", row_size);

    // Look at the first few rows to understand the pattern
    for poly_type_idx in 0..3 {
        for constraint_idx in 0..2.min(constraint_system.matrix.num_constraints) {
            let row_idx = poly_type_idx * constraint_system.matrix.num_constraints_padded + constraint_idx;
            let offset = row_idx * row_size;

            let row_data = &constraint_system.matrix.evaluations[offset..offset + row_size];
            let non_zero_count = row_data.iter().filter(|x| !x.is_zero()).count();

            println!(
                "Row {} (poly_type={}, constraint={}): {} non-zero values out of {}",
                row_idx, poly_type_idx, constraint_idx, non_zero_count, row_size
            );

            // Print first few values
            print!("  First 20 values: ");
            for i in 0..20.min(row_size) {
                if !row_data[i].is_zero() {
                    print!("X ");
                } else {
                    print!(". ");
                }
            }
            println!();
        }
    }

    // Test with specific evaluation points
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;

    // Create evaluation points
    let zs: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zx: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Debug: Test both variable orderings
    println!("\n=== Testing Variable Orderings ===");

    // Test 1: [zs, zx] ordering (s variables first, then x variables)
    let mut eval_point_sx = Vec::new();
    eval_point_sx.extend_from_slice(&zs);
    eval_point_sx.extend_from_slice(&zx);

    let eval_challenges_sx: Vec<<Fq as JoltField>::Challenge> = eval_point_sx
        .iter()
        .rev()  // Reverse for big-endian
        .map(|&x| x.into())
        .collect();

    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
    let sparse_eval_sx = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges_sx);

    // Test 2: [zx, zs] ordering (x variables first, then s variables)
    let mut eval_point_xs = Vec::new();
    eval_point_xs.extend_from_slice(&zx);
    eval_point_xs.extend_from_slice(&zs);

    let eval_challenges_xs: Vec<<Fq as JoltField>::Challenge> = eval_point_xs
        .iter()
        .rev()  // Reverse for big-endian
        .map(|&x| x.into())
        .collect();

    let sparse_eval_xs = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges_xs);

    println!("Sparse eval with [s, x] ordering: {:?}", sparse_eval_sx);
    println!("Sparse eval with [x, s] ordering: {:?}", sparse_eval_xs);

    // Compute dense evaluation using the bijection
    let eq_s_evals = EqPolynomial::<Fq>::evals(&zs);
    let eq_x_evals = EqPolynomial::<Fq>::evals(&zx);

    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);

    let dense_eval: Fq = (0..dense_size)
        .map(|i| {
            let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
            let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);

            // Debug specific values
            if i < 5 {
                println!(
                    "Dense index {}: maps to row={}, col={}, q_val={:?}",
                    i, row, col, dense_poly.Z[i]
                );
            }

            let eq_row = eq_s_evals[row];
            let eq_col = eq_x_evals[col];
            let q_val = dense_poly.Z[i];

            q_val * eq_row * eq_col
        })
        .sum();

    println!("\nDense eval: {:?}", dense_eval);

    // Compare evaluations
    println!("\n=== Comparison ===");
    println!("sparse_eval_sx == dense_eval: {}", sparse_eval_sx == dense_eval);
    println!("sparse_eval_xs == dense_eval: {}", sparse_eval_xs == dense_eval);

    if sparse_eval_sx != dense_eval && sparse_eval_xs != dense_eval {
        println!("\nERROR: Neither variable ordering matches!");

        // Debug: Manual computation for small example
        println!("\n=== Manual Debug Computation ===");

        // Check if the bijection is working correctly
        println!("Checking bijection for first few entries:");
        for i in 0..10.min(dense_size) {
            let row = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
            let col = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);

            // The row in the bijection corresponds to which polynomial
            // The col corresponds to the evaluation index within that polynomial

            // Map back to the original matrix position
            // We need to understand which polynomial type and constraint this comes from

            println!("Dense idx {} -> bijection row={}, col={}", i, row, col);
        }
    }
}