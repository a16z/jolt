use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum WrapperSpartanHyperKzgFactsError {
    #[error("R1CS dimension {dimension}={value} cannot be padded to a power of two")]
    DimensionOverflow {
        dimension: &'static str,
        value: usize,
    },
    #[error("public input layout overflow: start {start}, len {len}")]
    PublicInputLayoutOverflow { start: usize, len: usize },
    #[error("public input layout has total {total}, but segment ends at {end}")]
    PublicInputLayoutMismatch { total: usize, end: usize },
}
