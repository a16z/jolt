//! Stateless claim types for PCS operations.

use jolt_claims::EvaluationClaim;
use jolt_field::Field;
use jolt_poly::Polynomial;

/// Prover-side opening claim: polynomial, evaluation point, and claimed value.
#[derive(Clone, Debug)]
pub struct ProverOpeningClaim<F: Field, P = Polynomial<F>> {
    pub polynomial: P,
    pub evaluation: EvaluationClaim<F>,
}

/// Verifier-side opening claim: commitment, point, and claimed value.
#[derive(Clone, Debug)]
pub struct VerifierOpeningClaim<F: Field, C> {
    pub commitment: C,
    pub evaluation: EvaluationClaim<F>,
}

pub type ProverClaim<F, P = Polynomial<F>> = ProverOpeningClaim<F, P>;
pub type VerifierClaim<F, C> = VerifierOpeningClaim<F, C>;
