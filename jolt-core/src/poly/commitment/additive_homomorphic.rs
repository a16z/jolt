//! Trait for commitment schemes that support additive homomorphism.
use std::borrow::Borrow;

use super::commitment_scheme::CommitmentScheme;
use crate::utils::errors::ProofVerifyError;

/// A commitment scheme is additively homomorphic if given commitments to
/// polynomials p₁, p₂, ..., pₙ and scalars α₁, α₂, ..., αₙ, one can
/// produce evaluaton proof for a commitment to Σᵢ αᵢ · pᵢ.
pub trait AdditivelyHomomorphic: CommitmentScheme {
    /// Homomorphically combines multiple commitments into a single commitment,
    /// computed as a linear combination with the given coefficients.
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Result<Self::Commitment, ProofVerifyError>;

    /// Homomorphically combines multiple opening proof hints into a single hint,
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let _ = (hints, coeffs);
        unimplemented!("Hint combination not implemented for this commitment scheme")
    }
}
