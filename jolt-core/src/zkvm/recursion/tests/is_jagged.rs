//! Test to verify that the Dory sparse matrix is actually jagged.

use crate::transcripts::Transcript;
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
    zkvm::recursion::{
        bijection::{JaggedTransform, VarCountJaggedBijection},
        constraints_sys::{ConstraintSystem, ConstraintType},
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::Zero;
use serial_test::serial;

#[test]
#[serial]
fn test_dory_matrix_is_jagged() {
    use ark_ff::UniformRand;
    use rand::thread_rng;

    // Initialize Dory globals
    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let num_vars = 4;
    let mut rng = thread_rng();

    // Create a test Dory proof to get witness data
    let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
    let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

    let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
    let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

    let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
        .collect();

    let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
    let proof = DoryCommitmentScheme::prove(
        &prover_setup,
        &poly,
        &point,
        Some(hint),
        &mut prover_transcript,
    );

    let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
    let mut extract_transcript = crate::transcripts::Blake2bTranscript::new(b"test");

    let (constraint_system, _hints) = ConstraintSystem::new(
        &proof,
        &verifier_setup,
        &mut extract_transcript,
        &point,
        &evaluation,
        &commitment,
    )
    .expect("System creation should succeed");

    // Build dense polynomial and bijection
    let (_dense_poly, bijection, _mapping) = constraint_system.build_dense_polynomial();

    // Check jaggedness by scanning each row to find where it becomes all zeros
    let num_rows = constraint_system.matrix.num_rows;
    let row_size = 1 << constraint_system.matrix.num_constraint_vars;

    use std::collections::HashMap;
    let mut row_heights: HashMap<usize, usize> = HashMap::new();

    for row_idx in 0..num_rows {
        let offset = constraint_system.matrix.storage_offset(row_idx);

        // Find the height of this row
        let mut height = 0;
        for i in 0..row_size {
            let val = constraint_system.matrix.evaluations[offset + i];
            if !val.is_zero() {
                height = i + 1;
            }
        }

        if height == 0 {
            continue;
        }

        // Verify all values past the height are zero
        for i in height..row_size {
            let val = constraint_system.matrix.evaluations[offset + i];
            assert!(
                val.is_zero(),
                "Row {} should have all zeros past height {}, but found non-zero at position {}",
                row_idx, height, i
            );
        }

        *row_heights.entry(height).or_insert(0) += 1;
    }

    // Verify we have at least two different row heights (jaggedness)
    assert!(
        row_heights.len() >= 2,
        "Matrix should be jagged, but found only {} height(s)",
        row_heights.len()
    );

    // Verify compression ratio
    let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);
    let total_sparse_size = constraint_system.matrix.evaluations.len();
    let compression_ratio =
        (total_sparse_size - dense_size) as f64 / total_sparse_size as f64 * 100.0;

    assert!(
        compression_ratio > 50.0,
        "Compression ratio should be significant (>50%), but got {:.2}%",
        compression_ratio
    );
}
