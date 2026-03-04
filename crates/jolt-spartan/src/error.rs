//! Error types for Spartan proof generation and verification.

/// Errors that can occur during Spartan proving or verification.
///
/// Wraps errors from the underlying sumcheck and opening proof sub-protocols,
/// and adds Spartan-specific failure modes such as unsatisfied R1CS constraints
/// and evaluation mismatches.
#[derive(Debug, thiserror::Error)]
pub enum SpartanError {
    /// The witness does not satisfy the R1CS constraint system.
    ///
    /// The constraint at the given index has $Az_i \cdot Bz_i \neq Cz_i$.
    #[error("R1CS constraint violation at index {0}")]
    ConstraintViolation(usize),

    /// The sumcheck sub-protocol rejected the proof.
    #[error("sumcheck failed: {0}")]
    Sumcheck(#[from] jolt_sumcheck::SumcheckError),

    /// An opening proof failed verification.
    #[error("opening proof failed: {0}")]
    Opening(#[from] jolt_openings::OpeningsError),

    /// The final sumcheck evaluation does not match the expected value
    /// derived from the matrix MLEs and equality polynomial.
    #[error("evaluation mismatch: the sumcheck final value does not match the MLE evaluations")]
    EvaluationMismatch,
}
