use jolt_riscv::{SourceInstructionKind, TraceRowError};

use crate::preprocess::PreprocessingError;

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error(transparent)]
    Program(#[from] crate::ProgramError),
    #[error("Jolt program does not contain ELF bytes for the selected backend")]
    MissingElfBytes,
    #[error("execution backend failed: {0}")]
    Backend(&'static str),
    #[error(transparent)]
    InvalidRow(#[from] TraceRowError),
    #[error(transparent)]
    Preprocessing(#[from] PreprocessingError),
    #[error("cycle contains source-only instruction {0:?}")]
    SourceOnlyCycle(SourceInstructionKind),
    #[error("no bytecode PC for address {address:#x}, sequence {virtual_sequence_remaining:?}")]
    MissingBytecodePc {
        address: u64,
        virtual_sequence_remaining: Option<u16>,
    },
    #[error("bytecode PC {pc} does not fit u32")]
    BytecodePcTooWide { pc: usize },
    #[error("a partially consumed trace cannot be transferred to a retained witness")]
    PartiallyConsumed,
    #[error("field event at cycle {cycle} is out of range or not strictly ordered")]
    InvalidFieldEvent { cycle: usize },
    #[error("trace row {cycle} does not match its bytecode instruction")]
    BytecodeMismatch { cycle: usize },
}
