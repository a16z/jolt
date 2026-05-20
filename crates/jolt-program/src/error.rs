use jolt_riscv::SourceInstructionKind;

#[derive(Debug, thiserror::Error)]
pub enum ProgramError {
    #[error("unsupported program architecture: {0}")]
    UnsupportedArchitecture(&'static str),
    #[error("malformed program image: {0}")]
    MalformedImage(&'static str),
    #[error("source instruction is not legal in the selected profile: {0:?}")]
    IllegalSourceInstruction(SourceInstructionKind),
    #[error(transparent)]
    Expansion(#[from] crate::expand::ExpansionError),
}
