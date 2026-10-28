use common::jolt_device::MemoryLayoutError;
use jolt_riscv::{JoltInstructionRow, SourceInstructionKind, TraceRowError};

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
    #[error("field event at cycle {cycle} is out of range or not strictly ordered")]
    InvalidFieldEvent { cycle: usize },
    #[error("RAM minimum address exceeds maximum address")]
    InvalidRamBounds,
    #[error("executed instruction differs from bytecode at PC {pc}: expected {expected:?}, actual {actual:?}")]
    BytecodeInstructionMismatch {
        pc: u32,
        expected: Box<JoltInstructionRow>,
        actual: Box<JoltInstructionRow>,
    },
    #[error(transparent)]
    TraceAllocation(#[from] TraceAllocationError),
    #[error("an explicit trace allocation limit supports only serial eager tracing")]
    UnsupportedTraceAllocationMode,
    #[error(transparent)]
    InvalidMemoryLayout(#[from] MemoryLayoutError),
}

/// The trace-storage request an explicit trace allocation limit stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceAllocationPhase {
    /// Setup buffers, the initial row reserve and the pending cycle window.
    Initial,
    RowGrowth,
    /// Converted-chunk metadata and field-event buffers.
    Conversion,
    FieldMerge,
    FinalMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TraceAllocationError {
    #[error(
        "{phase:?} trace storage needs {required_bytes} bytes, over the {limit_bytes}-byte trace allocation limit"
    )]
    LimitExceeded {
        phase: TraceAllocationPhase,
        limit_bytes: usize,
        required_bytes: usize,
    },
    #[error("{phase:?} trace storage size overflows usize")]
    SizeOverflow { phase: TraceAllocationPhase },
    #[error("the allocator refused {phase:?} trace storage")]
    AllocationFailed { phase: TraceAllocationPhase },
}
