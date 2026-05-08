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
    #[error("malformed normalized instruction: {0}")]
    MalformedInstruction(&'static str),
    #[error("instruction {0:?} is not legal in finalized provider-free bytecode")]
    IllegalTargetInstruction(jolt_riscv::JoltInstructionKind),
    #[error("unsupported CSR 0x{0:03x}")]
    UnsupportedCsr(u16),
    #[error("unsupported instruction expansion")]
    UnsupportedInstruction,
    #[error("registered inline expansion requires an InlineExpansionProvider")]
    InlineProviderRequired,
}
