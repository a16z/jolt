//! Trait for commitment schemes that support additive homomorphism.
//!
//! This module provides the `AdditivelyHomomorphic` trait for commitment schemes
//! that allow homomorphic combination of commitments.

use std::borrow::Borrow;

use super::commitment_scheme::CommitmentScheme;
use crate::utils::errors::ProofVerifyError;

/// Trait for commitment schemes that support additive homomorphism.
///
/// A commitment scheme is additively homomorphic if given commitments to
/// polynomials p₁, p₂, ..., pₙ and scalars α₁, α₂, ..., αₙ, one can
/// efficiently compute a commitment to the polynomial Σᵢ αᵢ · pᵢ.
pub trait AdditivelyHomomorphic: CommitmentScheme {
    /// Homomorphically combines multiple commitments into a single commitment,
    /// computed as a linear combination with the given coefficients.
    ///
    /// # Arguments
    /// * `commitments` - The commitments to combine
    /// * `coeffs` - The coefficients for the linear combination
    ///
    /// # Returns
    /// The combined commitment C = Σᵢ coeffs[i] · commitments[i]
    ///
    /// # Errors
    /// Returns an error if the scheme doesn't support additive homomorphism
    /// or if the inputs are invalid (e.g., mismatched lengths).
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Result<Self::Commitment, ProofVerifyError>;

    /// Homomorphically combines multiple opening proof hints into a single hint,
    /// computed as a linear combination with the given coefficients.
    ///
    /// # Arguments
    /// * `hints` - The hints to combine
    /// * `coeffs` - The coefficients for the linear combination
    ///
    /// # Returns
    /// The combined hint, or a default/unimplemented value for schemes
    /// that don't use hints for homomorphic operations.
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let _ = (hints, coeffs); // Avoid unused parameter warnings
                                 // Default implementation for schemes where hint combination isn't needed
        unimplemented!("Hint combination not implemented for this commitment scheme")
    }
}
