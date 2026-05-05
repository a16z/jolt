#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PreprocessingError {
    #[error(
        "bytecode has non-decreasing inline sequences at index {bytecode_index} (address {address:#x}): previous max sequence {previous_max_sequence}, new sequence {new_sequence}"
    )]
    NonDecreasingInlineSequence {
        bytecode_index: usize,
        address: usize,
        previous_max_sequence: u16,
        new_sequence: u16,
    },
}
