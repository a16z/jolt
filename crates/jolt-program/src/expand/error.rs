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
    #[error("malformed normalized instruction: {0}")]
    MalformedInstruction(&'static str),
    #[error("unsupported CSR 0x{0:03x}")]
    UnsupportedCsr(u16),
    #[error("unsupported instruction expansion")]
    UnsupportedInstruction,
    #[error("registered inline expansion requires an InlineExpansionProvider")]
    InlineProviderRequired,
}
