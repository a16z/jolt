//! Fixed test for the jagged polynomial bijection relation

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
use rayon::prelude::*;

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
fn fixed_jagged_relation_dory_witness() {
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

    // Get the constraint system and build dense polynomial
    let constraint_system = &prover.constraint_system;

    // Build the bijection and dense polynomial
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);
    let (bijection, mapping) = builder.build();
    let (dense_poly, _, _) = constraint_system.build_dense_polynomial();

    // Get dimensions
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;

    // Create evaluation points
    let zs: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zx: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // The sparse matrix M(s, x) has variables ordered as [s, x]
    // So evaluation point should be [zs, zx]
    let mut eval_point = Vec::new();
    eval_point.extend_from_slice(&zs);
    eval_point.extend_from_slice(&zx);

    // Convert to challenges (reverse for big-endian)
    let eval_challenges: Vec<<Fq as JoltField>::Challenge> = eval_point
        .iter()
        .rev()
        .map(|&x| x.into())
        .collect();

    // Evaluate sparse polynomial
    let sparse_mlpoly = MultilinearPolynomial::from(constraint_system.matrix.evaluations.clone());
    let sparse_eval = PolynomialEvaluation::evaluate(&sparse_mlpoly, &eval_challenges);

    // Pre-compute eq evaluations
    let eq_s_evals = EqPolynomial::<Fq>::evals(&zs);
    let eq_x_evals = EqPolynomial::<Fq>::evals(&zx);

    // Compute dense evaluation using the correct mapping
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);

    let dense_eval: Fq = (0..dense_size)
        .into_par_iter()
        .map(|dense_idx| {
            // Get polynomial index and evaluation index from bijection
            let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, dense_idx);
            let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, dense_idx);

            // Decode to get constraint index and polynomial type
            let (constraint_idx, poly_type) = mapping.decode(poly_idx);

            // Get the matrix row index
            let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);

            // The key insight: matrix_row corresponds to the 's' coordinate in M(s,x)
            // and eval_idx corresponds to the 'x' coordinate

            // However, we need to ensure that indices beyond the actual matrix size
            // evaluate to zero (this is the padding)
            if matrix_row >= constraint_system.matrix.num_rows {
                // This shouldn't happen with a well-formed bijection
                return Fq::zero();
            }

            // Get eq evaluations
            let eq_s = if matrix_row < eq_s_evals.len() {
                eq_s_evals[matrix_row]
            } else {
                Fq::zero()
            };

            let eq_x = if eval_idx < eq_x_evals.len() {
                eq_x_evals[eval_idx]
            } else {
                Fq::zero()
            };

            // Get the dense polynomial value
            let q_val = dense_poly.Z[dense_idx];

            q_val * eq_s * eq_x
        })
        .sum();

    println!("Sparse eval: {:?}", sparse_eval);
    println!("Dense eval: {:?}", dense_eval);
    println!("Match: {}", sparse_eval == dense_eval);

    // Verify that sparse and dense evaluations match
    assert_eq!(
        sparse_eval, dense_eval,
        "Sparse and dense evaluations should match"
    );
}