//! End-to-end tests for the recursion SNARK using the unified API

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
            hyrax::Hyrax,
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::{
        recursion::{
            ConstraintType, RecursionProver, RecursionVerifier, RecursionVerifierInput,
        },
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::UniformRand;
use ark_grumpkin::Projective as GrumpkinProjective;
use ark_std::test_rng;
use dory::backends::arkworks::ArkGT;
use serial_test::serial;

#[test]
#[serial]
fn test_recursion_snark_e2e_with_dory() {
    use crate::poly::commitment::dory::wrappers::ArkDoryProof;

    // Initialize test environment
    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let mut rng = test_rng();

    // ============ CREATE A DORY PROOF TO VERIFY ============
    // Create test polynomial
    let num_vars = 4;
    let poly_coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly_coefficients));

    // Setup Dory
    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);

    // Commit to polynomial
    let (commitment, hint) =
        <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    // Create evaluation point using transcript challenges
    let mut point_transcript: Blake2bTranscript = Transcript::new(b"test_point");
    let point_challenges: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();

    // Compute evaluation
    let evaluation = PolynomialEvaluation::evaluate(&poly, &point_challenges);

    // Generate Dory proof
    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");
    let opening_proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &point_challenges,
        Some(hint),
        &mut prover_transcript,
    );

    // Convert to ArkDoryProof
    let ark_proof = ArkDoryProof::from(opening_proof);
    let ark_commitment = ArkGT::from(commitment);

    // ============ CREATE RECURSION PROVER FROM DORY PROOF ============
    // Generate gamma and delta for batching
    let gamma = Fq::rand(&mut rng);
    let delta = Fq::rand(&mut rng);

    // Create prover using Dory witness generation
    let mut witness_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");

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

    // Extract constraint information before moving prover
    let num_constraints = prover.constraint_system.num_constraints();
    let num_vars = prover.constraint_system.num_vars();
    let num_s_vars = prover.constraint_system.num_s_vars();
    let _matrix_num_vars = prover.constraint_system.matrix.num_vars;
    let num_constraint_vars = prover.constraint_system.matrix.num_constraint_vars;
    let num_constraints_padded = prover.constraint_system.matrix.num_constraints_padded;

    // Build dense polynomial, bijection, and mapping for Stage 3
    let (dense_poly, jagged_bijection, jagged_mapping) = prover.constraint_system.build_dense_polynomial();
    let dense_num_vars = dense_poly.get_num_vars();

    // Precompute matrix row indices for all polynomial indices
    let num_polynomials = jagged_bijection.num_polynomials();
    let mut matrix_rows = Vec::with_capacity(num_polynomials);
    for poly_idx in 0..num_polynomials {
        let (constraint_idx, poly_type) = jagged_mapping.decode(poly_idx);
        let matrix_row = prover.constraint_system.matrix.row_index(poly_type, constraint_idx);
        matrix_rows.push(matrix_row);
    }


    // Extract constraint types for verification
    let constraint_types: Vec<ConstraintType> = prover
        .constraint_system
        .constraints
        .iter()
        .map(|c| c.constraint_type.clone())
        .collect();

    let _ = (num_constraints, num_vars, num_s_vars);

    // Create transcript for proving
    let mut prover_transcript = Blake2bTranscript::new(b"recursion_snark");

    // Setup Hyrax PCS (which works with Fq)
    const RATIO: usize = 1;
    type HyraxPCS = Hyrax<RATIO, GrumpkinProjective>;

    let hyrax_prover_setup = <HyraxPCS as CommitmentScheme>::setup_prover(dense_num_vars);

    // Commit to the dense polynomial using Hyrax (after jagged transform)
    let dense_mlpoly = MultilinearPolynomial::from(dense_poly.Z);
    let (dense_commitment, _) =
        <HyraxPCS as CommitmentScheme>::commit(&dense_mlpoly, &hyrax_prover_setup);

    // Run the unified prover
    let recursion_proof = prover
        .prove_with_pcs::<Blake2bTranscript, HyraxPCS>(&mut prover_transcript, &hyrax_prover_setup)
        .expect("Failed to generate recursion proof");

    // ============ VERIFY THE RECURSION PROOF ============

    // Create verifier input
    let verifier_input = RecursionVerifierInput {
        constraint_types,
        num_vars,
        num_constraint_vars,
        num_s_vars,
        num_constraints,
        num_constraints_padded,
        jagged_bijection,
        jagged_mapping,
        matrix_rows,
    };

    // Create verifier
    let verifier = RecursionVerifier::<Fq>::new(verifier_input);

    // Create transcript for verification
    let mut verifier_transcript = Blake2bTranscript::new(b"recursion_snark");

    // Setup Hyrax verifier
    let hyrax_verifier_setup = <HyraxPCS as CommitmentScheme>::setup_verifier(&hyrax_prover_setup);

    // Verify the proof
    let verification_result = verifier
        .verify::<Blake2bTranscript, HyraxPCS>(
            &recursion_proof,
            &mut verifier_transcript,
            &dense_commitment,
            &hyrax_verifier_setup,
        )
        .expect("Verification should not fail");

    assert!(verification_result, "Recursion proof verification failed!");
}
