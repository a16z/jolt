#[derive(Debug, thiserror::Error)]
pub enum ProgramError {
    #[error("unsupported program architecture: {0}")]
    UnsupportedArchitecture(&'static str),
    #[error("malformed program image: {0}")]
    MalformedImage(&'static str),
}
