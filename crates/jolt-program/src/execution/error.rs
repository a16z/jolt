#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("executable program does not contain ELF bytes for the selected backend")]
    MissingElfBytes,
    #[error("execution backend failed: {0}")]
    Backend(&'static str),
}
