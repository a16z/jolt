//! Stateless claim types for PCS operations.

use jolt_field::Field;
use jolt_poly::Polynomial;

use crate::schemes::CommitmentSchemeVerifier;

/// Prover-side opening claim: polynomial, evaluation point, and claimed value.
#[derive(Clone, Debug)]
pub struct ProverClaim<F: Field> {
    pub polynomial: Polynomial<F>,
    pub point: Vec<F>,
    pub eval: F,
}

/// Verifier-side opening claim: commitment, evaluation point, and claimed value.
///
/// Generic over the verifier-side trait [`CommitmentSchemeVerifier`] only, so
/// verifier-only crates never depend on the prover surface.
#[derive(Clone, Debug)]
pub struct OpeningClaim<F: Field, PCS: CommitmentSchemeVerifier<Field = F>> {
    pub commitment: PCS::Output,
    pub point: Vec<F>,
    pub eval: F,
}
