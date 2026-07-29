#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PreprocessingError {
    #[error("bytecode instruction is not legal in the selected target profile: {0:?}")]
    IllegalTargetInstruction(jolt_riscv::JoltInstructionKind),
    #[error("bytecode instruction address is invalid for bytecode indexing: {0:#x}")]
    InvalidBytecodeAddress(usize),
    #[error(
        "store instruction at address {address:#x} writes rd x{rd}: the lattice fused-inc encoding requires store/rd-write disjointness"
    )]
    StoreWritesRd { address: usize, rd: u8 },
    #[error(
        "bytecode has invalid inline sequence at index {bytecode_index} (address {address:#x}): previous sequence {previous_sequence}, expected next sequence {expected_sequence}, new sequence {new_sequence}"
    )]
    InvalidInlineSequence {
        bytecode_index: usize,
        address: usize,
        previous_sequence: u16,
        expected_sequence: u16,
        new_sequence: u16,
    },
    #[cfg(feature = "field-inline")]
    #[error("invalid field-inline bytecode metadata: {0}")]
    InvalidFieldInlineMetadata(#[from] crate::field_inline::FieldInlineMetadataError),
}
