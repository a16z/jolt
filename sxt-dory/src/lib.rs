//! # Dory
//!
//! Dory is a polynomial commitment scheme with excellent asymptotic performance as well as
//! practical efficiency. It is based on the work of Jonathan Lee (<https://eprint.iacr.org/2020/1274.pdf>).
//!
//! This crate provides a Rust implementation of the commitment scheme, intended to be usable as a
//! building block for other zk/SNARK protocols.

use crate::arithmetic::{Field, Group, MultiScalarMul, Pairing};
use crate::error::DoryError;
use crate::toy_transcript::ToyTranscript;
use crate::transcript::Transcript;

// use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;

mod core;
mod error;
mod primitives;

pub mod curve;
pub mod vmv;
pub use core::*;
pub use primitives::*;
pub use vmv::*;

// =================================================================================================
//                                        Dory PCS API
// =================================================================================================

/// Generate prover and verifier setups for the polynomial commitment scheme
///
/// This function creates both the prover setup (with all necessary generators)
/// and the derived verifier setup (with precomputed verification elements).
///
/// If `srs_filename` is provided, it will attempt to load from that file first.
/// If the file doesn't exist or loading fails, it will generate new parameters
/// and save them to the specified file.
///
/// # Parameters
/// - `rng`: Random number generator for setup generation
/// - `max_log_n`: Maximum log of the polynomial size (n = 2^max_log_n)
///
/// # Returns
/// A tuple containing (ProverSetup, VerifierSetup)
pub fn setup<E: Pairing, R: RngCore>(
    rng: R,
    max_log_n: usize,
) -> (ProverSetup<E>, VerifierSetup<E>) {
    let prover_setup = ProverSetup::new(rng, max_log_n);
    let verifier_setup = prover_setup.to_verifier_setup();
    (prover_setup, verifier_setup)
}

/// Generate prover and verifier setups with optional SRS file loading/saving
///
/// If `srs_filename` is provided, it will attempt to load from that file first.
/// If the file doesn't exist or loading fails, it will generate new parameters
/// and save them to the specified file using the new combined format.
///
/// # Parameters
/// - `rng`: Random number generator for setup generation (used only if file doesn't exist)
/// - `max_log_n`: Maximum log of the polynomial size (n = 2^max_log_n)
/// - `srs_filename`: Optional filename to load/save SRS parameters
///
/// # Returns
/// A tuple containing (ProverSetup, VerifierSetup)
pub fn setup_with_srs_file<E: Pairing, R: RngCore>(
    rng: R,
    max_log_n: usize,
    srs_filename: Option<&str>,
) -> (ProverSetup<E>, VerifierSetup<E>)
where
{
    match srs_filename {
        Some(filename) => {
            println!("trying to get srs...");
            println!("About to call load_from_file...");

            // Try to load both prover and verifier setups from combined file
            match (
                ProverSetup::load_from_file(filename),
                VerifierSetup::load_from_file(filename),
            ) {
                (Ok(prover_setup), Ok(verifier_setup)) => {
                    println!("✓ Loaded existing combined SRS from {}", filename);
                    (prover_setup, verifier_setup)
                }
                (Ok(prover_setup), Err(_)) => {
                    // File exists but no verifier setup (legacy format), generate verifier
                    println!(
                        "✓ Loaded prover setup from {}, generating verifier setup...",
                        filename
                    );
                    let verifier_setup = prover_setup.to_verifier_setup();
                    (prover_setup, verifier_setup)
                }
                (Err(e), _) => {
                    // File doesn't exist or failed to load, generate new and save
                    println!("Load failed: {}", e);
                    println!(
                        "Generating new SRS for max_log_n = {} (this may take a while...)",
                        max_log_n
                    );
                    let prover_setup = ProverSetup::new(rng, max_log_n);
                    let verifier_setup = prover_setup.to_verifier_setup();

                    println!("✓ Generated new SRS, now saving combined format...");
                    if let Err(e) = prover_setup.save_combined_to_file(filename) {
                        println!(
                            "Warning: Failed to save combined SRS to {}: {}",
                            filename, e
                        );
                    }
                    (prover_setup, verifier_setup)
                }
            }
        }
        None => {
            println!("No filename provided, generating new SRS...");
            let prover_setup = ProverSetup::new(rng, max_log_n);
            let verifier_setup = prover_setup.to_verifier_setup();
            (prover_setup, verifier_setup)
        }
    }
}

/// Generate and save SRS parameters to disk with standard naming
///
/// Creates a file named `k_{max_log_n}.srs` containing both prover and verifier setup parameters.
///
/// # Parameters
/// - `rng`: Random number generator for setup generation
/// - `max_log_n`: Maximum log of the polynomial size (n = 2^max_log_n)
///
/// # Returns
/// The filename where the SRS was saved
pub fn generate_srs<E: Pairing, R: RngCore>(
    rng: R,
    max_log_n: usize,
) -> Result<String, Box<dyn std::error::Error>>
where
{
    let filename = format!("k_{}.srs", max_log_n);
    let prover_setup = ProverSetup::<E>::new(rng, max_log_n);
    prover_setup.save_combined_to_file(&filename)?;
    Ok(filename)
}

/// Commit to a multilinear polynomial
///
/// This is a wrapper around `compute_polynomial_commitment` that provides
/// a simple interface for committing to polynomial coefficients.
///
/// # Parameters
/// - `polynomial`: The polynomial to commit to
/// - `offset`: Starting offset in the coefficient array
/// - `sigma`: matrix to commit is of size 2^sigma
/// - `prover_setup`: The prover setup containing generators
///
/// # Returns
/// A commitment element in the target group GT
pub fn commit<E, M1, P>(
    polynomial: &P,
    offset: usize,
    sigma: usize,
    prover_setup: &ProverSetup<E>,
) -> (E::GT, Vec<E::G1>)
where
    E: Pairing,
    M1: MultiScalarMul<E::G1>,
    P: Polynomial<<E::G1 as Group>::Scalar, E::G1> + ?Sized + Sync,
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    compute_polynomial_commitment::<E, M1, P, <E::G1 as Group>::Scalar, E::G1>(
        polynomial,
        offset,
        sigma,
        prover_setup,
    )
}

/// Evaluate a polynomial at a point and produce a proof
///
/// This function creates an evaluation proof for a polynomial at a given point.
/// It returns both the evaluation result and the proof.
///
/// # Parameters
/// - `coeffs`: The polynomial coefficients
/// - `point`: The evaluation point (multilinear)
/// - `sigma`: matrix to commit is of size 2^sigma
/// - `prover_setup`: The prover setup containing generators
/// - `transcript`: The transcript for Fiat-Shamir transformation
///
/// # Returns
/// A tuple containing:
/// - The evaluation result (scalar field element)
/// - The proof builder containing the evaluation proof
pub fn evaluate<
    E: Pairing,
    T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
    M1: MultiScalarMul<E::G1>,
    M2: MultiScalarMul<E::G2>,
    P: Polynomial<<E::G1 as Group>::Scalar, E::G1> + ?Sized + Sync,
>(
    polynomial: &P,
    row_commitments: Option<Vec<E::G1>>,
    point: &[<E::G1 as Group>::Scalar],
    sigma: usize,
    prover_setup: &ProverSetup<E>,
    transcript: T,
) -> DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
    <E::G1 as Group>::Scalar: Field + Clone,
{
    // Create the evaluation proof
    create_evaluation_proof::<E, T, M1, M2, P>(
        transcript,
        polynomial,
        row_commitments,
        point,
        sigma,
        prover_setup,
    )
}

/// Verify an evaluation proof
///
/// This function verifies that a polynomial committed to by `commitment`
/// evaluates to `evaluation` at the given `point`.
///
/// # Parameters
/// - `commitment`: The polynomial commitment in GT
/// - `evaluation`: The claimed evaluation result
/// - `point`: The evaluation point (multilinear)
/// - `proof`: The evaluation proof to verify
/// - `sigma`: matrix to commit is of size 2^sigma
/// - `verifier_setup`: The verifier setup containing verification elements
/// - `transcript`: Fresh transcript for verification
///
/// # Returns
/// `Ok(())` if verification succeeds, `Err(DoryError)` if it fails
pub fn verify<
    E: Pairing,
    T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
    M1: MultiScalarMul<E::G1>,
    M2: MultiScalarMul<E::G2>,
    MGT: MultiScalarMul<E::GT>,
>(
    commitment: E::GT,
    evaluation: <E::G1 as Group>::Scalar,
    point: &[<E::G1 as Group>::Scalar],
    proof: DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
    sigma: usize,
    verifier_setup: &VerifierSetup<E>,
    transcript: T,
) -> Result<(), DoryError>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
    <E::G1 as Group>::Scalar: Field,
{
    // Prepare verification data
    let commitment_batch = vec![commitment];
    let batching_factors = vec![<E::G1 as Group>::Scalar::one()];
    let evaluations = vec![evaluation];

    // Verify the proof
    verify_evaluation_proof::<E, T, M1, M2, MGT>(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        point,
        sigma,
        verifier_setup,
        transcript,
    )
}

/// Convenience test function to create a transcript for evaluation
///
/// # Parameters
/// - `domain`: Domain separator bytes for the transcript
///
/// # Returns
/// A new transcript instance using Blake2s256 hasher
pub fn create_transcript(domain: &[u8]) -> ToyTranscript {
    ToyTranscript::new(domain)
}
