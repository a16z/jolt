#[derive(Debug, thiserror::Error)]
pub enum ExpansionError {
    #[error("unsupported instruction expansion")]
    UnsupportedInstruction,
}
