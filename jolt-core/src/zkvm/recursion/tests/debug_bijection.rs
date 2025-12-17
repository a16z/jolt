//! Debug the bijection mapping to understand the mismatch

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::recursion::{
        bijection::{ConstraintSystemJaggedBuilder, JaggedTransform, VarCountJaggedBijection},
        RecursionProver,
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::{UniformRand, Zero};
use ark_std::test_rng;

#[test]
fn debug_bijection_mapping() {
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

    // Build the bijection directly to understand the mapping
    let builder = ConstraintSystemJaggedBuilder::from_constraints(&constraint_system.constraints);

    println!("=== Bijection Mapping Debug ===");
    println!("Number of polynomials in builder: {}", builder.polynomials.len());
    println!("First few polynomial mappings:");
    for (i, (constraint_idx, poly_type, num_vars)) in builder.polynomials.iter().take(10).enumerate() {
        println!(
            "  Poly {}: constraint_idx={}, poly_type={:?}, num_vars={}",
            i, constraint_idx, poly_type, num_vars
        );
    }

    let (bijection, mapping) = builder.build();

    // Now understand how the dense polynomial extracts values
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);
    println!("\nDense size: {}", dense_size);

    // Check specific mappings
    println!("\nChecking specific dense index mappings:");

    // For the first polynomial (which should be 4-var)
    for i in 0..20 {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, i);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, i);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);

        // Get the actual row in the matrix
        let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);
        let offset = constraint_system.matrix.storage_offset(matrix_row);

        // The value at this position in the sparse matrix
        let sparse_value = constraint_system.matrix.evaluations[offset + eval_idx];

        println!(
            "Dense idx {}: poly_idx={}, eval_idx={} -> constraint={}, type={:?}, matrix_row={}, sparse_val={:?}",
            i, poly_idx, eval_idx, constraint_idx, poly_type, matrix_row, sparse_value
        );

        // Verify this matches what build_dense_polynomial would extract
        if i < 16 {
            // For 4-var polynomials, the first 16 values should be non-zero
            assert!(!sparse_value.is_zero(), "Expected non-zero value at index {}", i);
        }
    }

    // Now check the actual dense polynomial built by the system
    let (dense_poly, _, _) = constraint_system.build_dense_polynomial();

    println!("\n=== Comparing extracted values ===");
    for i in 0..20 {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, i);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, i);

        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let matrix_row = constraint_system.matrix.row_index(poly_type, constraint_idx);
        let offset = constraint_system.matrix.storage_offset(matrix_row);

        let sparse_value = constraint_system.matrix.evaluations[offset + eval_idx];
        let dense_value = dense_poly.Z[i];

        println!(
            "Index {}: sparse={:?}, dense={:?}, match={}",
            i, sparse_value, dense_value, sparse_value == dense_value
        );

        if sparse_value != dense_value {
            println!("  MISMATCH at index {}!", i);
        }
    }
}