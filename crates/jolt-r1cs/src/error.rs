//! Error types for R1CS constraint systems.

/// Errors related to R1CS constraint satisfaction.
#[derive(Debug, thiserror::Error)]
pub enum R1csError {
    /// The witness does not satisfy the R1CS constraint system.
    ///
    /// The constraint at the given index has $Az_i \cdot Bz_i \neq Cz_i$.
    #[error("R1CS constraint violation at index {0}")]
    ConstraintViolation(usize),

    /// The witness does not satisfy the relaxed R1CS equation.
    ///
    /// The constraint at the given index has $Az_i \cdot Bz_i \neq u \cdot Cz_i + E_i$.
    #[error("relaxed R1CS constraint violation at index {0}: Az∘Bz ≠ u·Cz + E")]
    RelaxedConstraintViolation(usize),
}
