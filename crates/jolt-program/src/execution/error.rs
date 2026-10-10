use super::TraceRowError;
use jolt_riscv::SourceInstructionKind;

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
    SourceTrace(#[from] SourceTraceError),
}

/// Unsupported architectural transitions, rejected before the failing instruction
/// executes. Fetch, instruction kind, alignment, device cells, and text stores
/// are checked in that order; the text span is bounded before emulator setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SourceTraceError {
    #[error("unsupported source instruction {kind:?} at {pc:#x}")]
    UnsupportedInstruction {
        pc: u64,
        kind: SourceInstructionKind,
    },
    #[error("source PC {pc:#x} is outside decoded program instructions")]
    PcOutsideProgram { pc: u64 },
    #[error("misaligned source access at PC {pc:#x}: address {address:#x}, width {width}")]
    MisalignedAccess { pc: u64, address: u64, width: u8 },
    #[error("source store at PC {pc:#x} overlaps program text at {address:#x}")]
    StoreToProgramText { pc: u64, address: u64 },
    #[error("source access at PC {pc:#x} cannot access device cell at {address:#x}")]
    DeviceRegisterAccess { pc: u64, address: u64 },
    #[error("source program text span {span} bytes exceeds the slot table limit")]
    ProgramTextTooLarge { span: u64 },
}
