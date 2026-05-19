#[derive(Debug, thiserror::Error)]
pub enum ExpansionError {
    #[error("no virtual registers available in {pool} pool")]
    VirtualRegisterExhausted { pool: &'static str },
    #[error("virtual register {register} is outside the allocator range")]
    InvalidVirtualRegister { register: u8 },
    #[error("virtual register {register} was released before it was allocated")]
    UnallocatedVirtualRegister { register: u8 },
    #[error("inline virtual registers must be released before inline finalization")]
    InlineRegistersStillAllocated,
    #[error("inline expansion attempted to write to register {register}, but inline outputs must use x0 or virtual registers >= {minimum_register}")]
    InvalidInlineWriteTarget { register: u8, minimum_register: u8 },
    #[error("expansion produced an empty instruction sequence")]
    EmptySequence,
    #[error("expansion produced {actual} rows, exceeding the per-source capacity {capacity}")]
    CapacityExceeded { actual: usize, capacity: usize },
    #[error("expansion recursion depth exceeded {max_depth}")]
    RecursionDepthExceeded { max_depth: usize },
    #[error("temporary expansion register {index} was used before allocation")]
    UnallocatedTemporaryRegister { index: usize },
    #[error("temporary expansion register {index} was allocated more than once")]
    DuplicateTemporaryRegister { index: usize },
    #[error("temporary expansion register {index} was allocated but not released")]
    LeakedTemporaryRegister { index: usize },
    #[error("expansion allocated too many temporary registers: {actual}")]
    TooManyTemporaryRegisters { actual: usize },
    #[error("malformed Jolt row: {0}")]
    MalformedInstruction(&'static str),
    #[error("source instruction {0:?} has no direct final Jolt row")]
    IllegalSourceInstruction(jolt_riscv::SourceInstructionKind),
    #[error("instruction {0:?} is not legal in finalized provider-free bytecode")]
    IllegalTargetInstruction(jolt_riscv::JoltInstructionKind),
    #[error("unsupported CSR 0x{0:03x}")]
    UnsupportedCsr(u16),
    #[error("unsupported instruction expansion")]
    UnsupportedInstruction,
    #[error("registered inline {name} is internal-only and cannot be used as raw guest bytecode: {reason}")]
    InternalOnlyInline {
        name: &'static str,
        reason: &'static str,
    },
    #[error("registered inline expansion requires an InlineExpansionProvider")]
    InlineProviderRequired,
}
