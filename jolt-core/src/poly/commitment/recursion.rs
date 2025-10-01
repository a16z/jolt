//! Trait extensions for recursion-specific commitment scheme functionality.
//!
//! This module provides traits that extend the base `CommitmentScheme` trait
//! with additional functionality needed for recursive SNARK composition.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use super::commitment_scheme::CommitmentScheme;

/// Extension trait for commitment schemes that support recursion.
///
/// This trait provides additional functionality needed for recursive SNARK composition,
/// including precomputation of combined commitments and hints for efficient verification.
#[cfg(feature = "recursion")]
pub trait RecursionCommitmentScheme: CommitmentScheme {
    /// Precomputed data for efficient combined commitment verification in recursion mode.
    type CombinedCommitmentHint: Sync
        + Send
        + Clone
        + Debug
        + Default
        + CanonicalSerialize
        + CanonicalDeserialize;

    /// Precomputes the combined commitment and hint for recursion mode.
    ///
    /// This method allows the prover to precompute expensive operations
    /// (like GT scalar multiplications in pairing-based schemes) that can
    /// be used to speed up verification in recursive contexts.
    ///
    /// # Arguments
    /// * `commitments` - The commitments to combine
    /// * `coeffs` - The coefficients for the linear combination
    ///
    /// # Returns
    /// A tuple containing the combined commitment and a hint for verification
    fn precompute_combined_commitment<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> (Self::Commitment, Self::CombinedCommitmentHint);

    /// Homomorphically combines multiple commitments using a precomputed hint.
    ///
    /// # Arguments
    /// * `commitments` - The commitments to combine
    /// * `coeffs` - The coefficients for the linear combination
    /// * `hint` - Optional precomputed hint from `precompute_combined_commitment`
    ///
    /// # Returns
    /// The combined commitment
    fn combine_commitments_with_hint<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
        hint: Option<&Self::CombinedCommitmentHint>,
    ) -> Self::Commitment;
}

/// Auxiliary data for verifier assistance in recursion mode.
///
/// This trait provides a way for commitment schemes to pass additional
/// data from the prover to the verifier that can help with efficient
/// verification in recursive contexts.
#[cfg(feature = "recursion")]
pub trait RecursionAuxiliaryData: CommitmentScheme {
    /// Optional auxiliary data computed by the prover to help the verifier.
    /// Schemes that don't use auxiliary data should set this to ().
    type AuxiliaryVerifierData: Default
        + Debug
        + Sync
        + Send
        + CanonicalSerialize
        + CanonicalDeserialize
        + Clone;
}
