pub mod backend;
pub mod error;
pub mod trace;

use crate::ProgramError;
#[cfg(feature = "image")]
use crate::{
    expand::{expand_program, expand_program_with_provider, InlineExpansionProvider},
    image::decode_elf,
};
#[cfg(feature = "image")]
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow, RV64IMAC_JOLT};

pub use backend::{ExecutionBackend, TraceSource};
pub use error::TraceError;
pub use trace::{
    JoltProgram, MemoryImage, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead,
    RegisterState, RegisterWrite, TraceInputs, TraceOutput, TraceRow,
};

#[cfg(feature = "image")]
pub fn build_jolt_program(elf_bytes: &[u8]) -> Result<JoltProgram, ProgramError> {
    let image = decode_elf(elf_bytes, RV64IMAC_JOLT)?;
    let expanded_bytecode = expand_program(&image.instructions, RV64IMAC_JOLT)?
        .into_iter()
        .map(JoltInstructionRow::from)
        .collect();
    Ok(JoltProgram::from_rv64_image(
        elf_bytes.to_vec(),
        expanded_bytecode,
        image,
    ))
}

#[cfg(feature = "image")]
pub fn build_jolt_program_with_inline_provider<P: InlineExpansionProvider + ?Sized>(
    elf_bytes: &[u8],
    inline_provider: &mut P,
    profile: JoltInstructionProfile,
) -> Result<JoltProgram, ProgramError> {
    let image = decode_elf(elf_bytes, profile)?;
    let expanded_bytecode =
        expand_program_with_provider(&image.instructions, inline_provider, profile)?
            .into_iter()
            .map(JoltInstructionRow::from)
            .collect();
    Ok(JoltProgram::from_rv64_image(
        elf_bytes.to_vec(),
        expanded_bytecode,
        image,
    ))
}

#[cfg(not(feature = "image"))]
pub fn build_jolt_program(elf_bytes: &[u8]) -> Result<JoltProgram, ProgramError> {
    let _ = elf_bytes;
    Err(ProgramError::MalformedImage(
        "building a Jolt program from ELF bytes requires the jolt-program image feature",
    ))
}

#[cfg(not(feature = "image"))]
pub fn build_jolt_program_with_inline_provider<
    P: crate::expand::InlineExpansionProvider + ?Sized,
>(
    elf_bytes: &[u8],
    inline_provider: &mut P,
    profile: jolt_riscv::JoltInstructionProfile,
) -> Result<JoltProgram, ProgramError> {
    let _ = (elf_bytes, inline_provider, profile);
    Err(ProgramError::MalformedImage(
        "building a Jolt program from ELF bytes requires the jolt-program image feature",
    ))
}
