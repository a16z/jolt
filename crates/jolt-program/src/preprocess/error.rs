#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PreprocessingError {
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
}
