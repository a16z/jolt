pub mod allocator;
pub mod assembler;
pub mod error;
pub mod metadata;
pub mod sequences;

pub use allocator::ExpansionAllocator;
pub use error::ExpansionError;

use jolt_riscv::NormalizedInstruction;

pub fn expand_instruction(
    instruction: &NormalizedInstruction,
    _allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    Ok(vec![*instruction])
}

pub fn expand_program(
    instructions: impl IntoIterator<Item = NormalizedInstruction>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut allocator = ExpansionAllocator;
    let mut expanded = Vec::new();
    for instruction in instructions {
        expanded.extend(expand_instruction(&instruction, &mut allocator)?);
    }
    Ok(expanded)
}
