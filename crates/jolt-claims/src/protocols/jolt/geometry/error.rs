use thiserror::Error;

// The point error is protocol-neutral (field_inline geometry raises it too), so
// it lives in the crate's framework half; re-exported here at its historical path.
pub use crate::formula_error::JoltFormulaPointError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
pub enum JoltFormulaDimensionsError {
    #[error("{name} must be nonzero")]
    Zero { name: &'static str },
    #[error("{name} overflowed")]
    Overflow { name: &'static str },
    #[error(
        "lookup_virtual_chunk_bits ({lookup_virtual_chunk_bits}) must be >= committed_chunk_bits ({committed_chunk_bits})"
    )]
    InvalidChunkOrder {
        committed_chunk_bits: usize,
        lookup_virtual_chunk_bits: usize,
    },
    #[error("{name} underflowed")]
    Underflow { name: &'static str },
    #[error("{value_name} ({value}) must be divisible by {divisor_name} ({divisor})")]
    NotDivisible {
        value_name: &'static str,
        value: usize,
        divisor_name: &'static str,
        divisor: usize,
    },
    #[error("{value_name} ({value}) must be at most {max_name} ({max})")]
    Exceeds {
        value_name: &'static str,
        value: usize,
        max_name: &'static str,
        max: usize,
    },
    #[error("phase1_num_rounds ({phase1_num_rounds}) must be <= log_t ({log_t})")]
    InvalidPhaseRounds {
        phase1_num_rounds: usize,
        log_t: usize,
    },
}

/// Require at least `expected` leading entries; callers consume the slice by
/// prefix, so longer inputs are accepted.
pub(crate) fn require_len<F>(values: &[F], expected: usize) -> Result<(), JoltFormulaPointError> {
    if values.len() < expected {
        return Err(JoltFormulaPointError::ChallengeLengthMismatch {
            expected,
            got: values.len(),
        });
    }
    Ok(())
}

/// [`require_len`] with the opening-point flavored error, for inputs that are
/// opening points rather than challenge vectors.
pub(crate) fn require_opening_point_len<F>(
    values: &[F],
    expected: usize,
) -> Result<(), JoltFormulaPointError> {
    if values.len() < expected {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected,
            got: values.len(),
        });
    }
    Ok(())
}
