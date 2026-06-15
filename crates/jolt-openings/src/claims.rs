//! Stateless claim types for PCS operations.

use jolt_field::Field;
use jolt_poly::{Point, Polynomial, HIGH_TO_LOW};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationClaim<F> {
    pub point: Point<HIGH_TO_LOW, F>,
    pub value: F,
}

impl<F> EvaluationClaim<F> {
    pub fn new(point: impl Into<Point<HIGH_TO_LOW, F>>, value: F) -> Self {
        Self {
            point: point.into(),
            value,
        }
    }
}

/// Prover-side opening claim: polynomial, commitment, evaluation point, and claimed value.
#[derive(Clone, Debug)]
pub struct ProverOpeningClaim<F: Field, C, P = Polynomial<F>> {
    pub polynomial: P,
    pub commitment: C,
    pub evaluation: EvaluationClaim<F>,
}

/// Verifier-side opening claim: commitment, point, and claimed value.
#[derive(Clone, Debug)]
pub struct VerifierOpeningClaim<F: Field, C> {
    pub commitment: C,
    pub evaluation: EvaluationClaim<F>,
}
