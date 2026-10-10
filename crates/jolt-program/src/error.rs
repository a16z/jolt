use jolt_riscv::SourceInstructionKind;

#[derive(Debug, thiserror::Error)]
pub enum ProgramError {
    #[error("unsupported program architecture: {0}")]
    UnsupportedArchitecture(&'static str),
    #[error("malformed program image: {0}")]
    MalformedImage(&'static str),
    #[error("source instruction is not legal in the selected profile: {0:?}")]
    IllegalSourceInstruction(SourceInstructionKind),
    #[error("compressed instruction at {address:#x} is not legal in the selected profile")]
    IllegalCompressedInstruction { address: u64 },
    #[error("the selected decode mode is not defined for a profile with compressed instructions")]
    DecodeModeUnsupportedByProfile,
    #[error(transparent)]
    Expansion(#[from] crate::expand::ExpansionError),
}
