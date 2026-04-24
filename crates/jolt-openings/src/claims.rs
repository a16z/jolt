//! Stateless claim types for PCS operations.

use jolt_field::Field;
use jolt_poly::Polynomial;

/// Prover-side opening claim: polynomial, evaluation point, and claimed value.
#[derive(Clone, Debug)]
pub struct ProverClaim<F: Field, P = Polynomial<F>> {
    pub polynomial: P,
    pub point: Vec<F>,
    pub eval: F,
}

/// Verifier-side opening claim: commitment, point, and claimed value.
#[derive(Clone, Debug)]
pub struct VerifierClaim<F: Field, C> {
    pub commitment: C,
    pub point: Vec<F>,
    pub eval: F,
}
