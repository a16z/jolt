//! Trait extensions for recursion-specific commitment scheme machinery.
//!
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use super::commitment_scheme::CommitmentScheme;

/// This trait provides additional functionality needed for recursive SNARK composition,
/// including precomputation of combined commitments and hints for efficient verification.
pub trait RecursionCommitmentScheme: CommitmentScheme {
    /// Precomputed data for efficient combined commitment verification in recursion mode.
    type CombinedCommitmentHint: Sync
        + Send
        + Clone
        + Debug
        + Default
        + CanonicalSerialize
        + CanonicalDeserialize;

    /// Auxiliary data computed by the prover to help the verifier.
    type AuxiliaryVerifierData: Default
        + Debug
        + Sync
        + Send
        + CanonicalSerialize
        + CanonicalDeserialize
        + Clone;

    /// Precomputes the combined commitment and hint for recursion mode.
    ///
    /// This method allows the prover to precompute expensive operations
    /// (like GT scalar multiplications in pairing-based schemes) that can
    /// be used to speed up verification in recursive contexts.
    fn precompute_combined_commitment<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> (Self::Commitment, Self::CombinedCommitmentHint);

    /// Homomorphically combines multiple commitments using a precomputed hint.
    fn combine_commitments_with_hint<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
        hint: Option<&Self::CombinedCommitmentHint>,
    ) -> Self::Commitment;

    /// Generates a proof of evaluation with auxiliary data for recursion.
    ///
    /// This is the recursion-aware version of `CommitmentScheme::prove()` that
    /// additionally returns auxiliary data needed for recursive verification.
    ///
    /// # Arguments
    /// * `setup` - The prover setup for the commitment scheme
    /// * `poly` - The multilinear polynomial being proved
    /// * `opening_point` - The point at which the polynomial is evaluated
    /// * `hint` - A hint that helps optimize the proof generation
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    ///
    /// # Returns
    /// A tuple containing the proof and auxiliary verifier data
    fn prove_with_auxiliary<ProofTranscript: crate::transcripts::Transcript>(
        setup: &Self::ProverSetup,
        poly: &crate::poly::multilinear_polynomial::MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        hint: Self::OpeningProofHint,
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Self::AuxiliaryVerifierData) {
        // Default implementation just calls the standard prove and returns default auxiliary data
        let proof = Self::prove(setup, poly, opening_point, hint, transcript);
        (proof, Self::AuxiliaryVerifierData::default())
    }
}
