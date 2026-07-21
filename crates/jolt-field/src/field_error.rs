/// Errors produced by backend-independent field helper algorithms.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FieldError {
    /// A caller supplied values with an invalid shape.
    #[error("invalid field input: {0}")]
    InvalidInput(String),

    /// A caller supplied a slice with an unexpected length.
    #[error("invalid field input size: expected {expected}, got {actual}")]
    InvalidSize {
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
}
