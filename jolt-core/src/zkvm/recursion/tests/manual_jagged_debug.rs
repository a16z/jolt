//! Manual debug of jagged relation to understand the exact issue

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
        bijection::{ConstraintSystemJaggedBuilder, JaggedTransform, VarCountJaggedBijection},
        RecursionProver,
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::{One, UniformRand, Zero};
use ark_std::test_rng;

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
fn manual_jagged_debug() {
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

    // Build bijection and dense polynomial
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);
    let (bijection, mapping) = builder.build();
    let (dense_poly, _, _) = constraint_system.build_dense_polynomial();

    // Get dimensions
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;

    println!("=== Debug Info ===");
    println!("num_s_vars: {}", num_s_vars);
    println!("num_x_vars: {}", num_x_vars);
    println!("num_rows: {}", constraint_system.matrix.num_rows);
    println!("row_size: {}", 1 << num_x_vars);

    // Create evaluation points
    let zs: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zx: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Manually compute sparse evaluation to understand the exact formula
    println!("\n=== Manual Sparse Evaluation ===");

    // The sparse polynomial M(s, x) is evaluated as:
    // sum_{s,x in {0,1}^n} M[s][x] * eq(s, zs) * eq(x, zx)

    let mut manual_sparse_eval = Fq::zero();
    let row_size = 1 << num_x_vars;

    // Pre-compute eq evaluations
    let eq_s_evals = EqPolynomial::<Fq>::evals(&zs);
    let eq_x_evals = EqPolynomial::<Fq>::evals(&zx);

    for s_idx in 0..constraint_system.matrix.num_rows {
        for x_idx in 0..row_size {
            let flat_idx = s_idx * row_size + x_idx;
            let val = constraint_system.matrix.evaluations[flat_idx];

            if !val.is_zero() && s_idx < 10 && x_idx < 5 {
                println!(
                    "  Non-zero at s={}, x={}: val={:?}",
                    s_idx, x_idx, val
                );
            }

            let eq_s = eq_s_evals[s_idx];
            let eq_x = eq_x_evals[x_idx];

            manual_sparse_eval += val * eq_s * eq_x;
        }
    }

    println!("Manual sparse eval: {:?}", manual_sparse_eval);

    // Now compute using MultilinearPolynomial evaluation
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zs);
    eval_point.extend_from_slice(&zx);

    let eval_challenges: Vec<<Fq as JoltField>::Challenge> = eval_point
        .iter()
        .rev()
        .map(|&x| x.into())
        .collect();

    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

    println!("Sparse eval via MLP: {:?}", sparse_eval);
    println!("Match: {}", manual_sparse_eval == sparse_eval);

    // Now debug the dense evaluation
    println!("\n=== Dense Evaluation Debug ===");

    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);
    println!("Dense size: {}", dense_size);

    // Check first few mappings
    for i in 0..10.min(dense_size) {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, i);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, i);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);

        // Get the value from sparse matrix
        let sparse_flat_idx = matrix_row * row_size + eval_idx;
        let sparse_val = constraint_system.matrix.evaluations[sparse_flat_idx];
        let dense_val = dense_poly.Z[i];

        println!(
            "Dense idx {}: poly_idx={}, eval_idx={} -> matrix_row={}, sparse_val={:?}, dense_val={:?}, match={}",
            i, poly_idx, eval_idx, matrix_row, sparse_val, dense_val, sparse_val == dense_val
        );
    }

    // Compute dense evaluation
    let mut dense_eval = Fq::zero();
    for i in 0..dense_size {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, i);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, i);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);

        let eq_s = eq_s_evals[matrix_row];
        let eq_x = eq_x_evals[eval_idx];
        let q_val = dense_poly.Z[i];

        dense_eval += q_val * eq_s * eq_x;
    }

    println!("\nDense eval: {:?}", dense_eval);
    println!("Sparse eval: {:?}", sparse_eval);
    println!("Match: {}", dense_eval == sparse_eval);
}