#[derive(Debug, thiserror::Error)]
pub enum PreprocessingError {
    #[error("invalid program preprocessing input")]
    InvalidInput,
}
