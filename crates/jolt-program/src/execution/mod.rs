pub mod backend;
pub mod error;
pub mod trace;

use crate::ProgramError;
#[cfg(feature = "image")]
use crate::{expand::expand_program, image::decode_elf};

pub use backend::{ExecutionBackend, TraceSource};
pub use error::TraceError;
pub use trace::{
    ExecutableProgram, MemoryImage, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead,
    RegisterState, RegisterWrite, TraceInputs, TraceOutput, TraceRow,
};

#[cfg(feature = "image")]
pub fn build_executable(elf_bytes: &[u8]) -> Result<ExecutableProgram, ProgramError> {
    let image = decode_elf(elf_bytes)?;
    let expanded_bytecode = expand_program(image.instructions.iter().copied())?;
    Ok(ExecutableProgram::from_parts(
        elf_bytes.to_vec(),
        expanded_bytecode,
        image.memory_init,
        image.program_end,
        image.entry_address,
    ))
}

#[cfg(not(feature = "image"))]
pub fn build_executable(elf_bytes: &[u8]) -> Result<ExecutableProgram, ProgramError> {
    let _ = elf_bytes;
    Err(ProgramError::MalformedImage(
        "building an executable from ELF bytes requires the jolt-program image feature",
    ))
}
