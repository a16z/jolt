use thiserror::Error;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
pub enum DoryAssistFormulaPointError {
    #[error("challenge length mismatch: expected {expected}, got {got}")]
    ChallengeLengthMismatch { expected: usize, got: usize },
    #[error("opening point length mismatch: expected {expected}, got {got}")]
    OpeningPointLengthMismatch { expected: usize, got: usize },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
pub enum DoryAssistFormulaDimensionsError {
    #[error("{name} must be nonzero")]
    Zero { name: &'static str },
    #[error("{name} overflowed")]
    Overflow { name: &'static str },
    #[error("packed variables ({packed_vars}) must be >= polynomial variables ({poly_vars})")]
    InvalidPackingPrefix {
        packed_vars: usize,
        poly_vars: usize,
    },
}
