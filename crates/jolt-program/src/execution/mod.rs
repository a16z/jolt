pub mod backend;
pub mod error;
pub mod trace;

use crate::ProgramError;
#[cfg(feature = "image")]
use crate::{
    expand::{expand_program, expand_program_with_provider, InlineExpansionProvider},
    image::decode_elf,
};

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
    Ok(ExecutableProgram::from_decoded_image(
        elf_bytes.to_vec(),
        expanded_bytecode,
        image,
    ))
}

#[cfg(feature = "image")]
pub fn build_executable_with_inline_provider<P: InlineExpansionProvider + ?Sized>(
    elf_bytes: &[u8],
    inline_provider: &mut P,
) -> Result<ExecutableProgram, ProgramError> {
    let image = decode_elf(elf_bytes)?;
    let expanded_bytecode =
        expand_program_with_provider(image.instructions.iter().copied(), inline_provider)?;
    Ok(ExecutableProgram::from_decoded_image(
        elf_bytes.to_vec(),
        expanded_bytecode,
        image,
    ))
}

#[cfg(not(feature = "image"))]
pub fn build_executable(elf_bytes: &[u8]) -> Result<ExecutableProgram, ProgramError> {
    let _ = elf_bytes;
    Err(ProgramError::MalformedImage(
        "building an executable from ELF bytes requires the jolt-program image feature",
    ))
}

#[cfg(not(feature = "image"))]
pub fn build_executable_with_inline_provider<P: crate::expand::InlineExpansionProvider + ?Sized>(
    elf_bytes: &[u8],
    inline_provider: &mut P,
) -> Result<ExecutableProgram, ProgramError> {
    let _ = (elf_bytes, inline_provider);
    Err(ProgramError::MalformedImage(
        "building an executable from ELF bytes requires the jolt-program image feature",
    ))
}
