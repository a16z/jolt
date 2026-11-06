pub mod backend;
pub mod error;
pub mod trace;

use crate::ProgramError;
#[cfg(feature = "image")]
use crate::{
    expand::{InlineExpansionProvider, NoInlineExpansionProvider},
    image::decode_elf,
};
#[cfg(feature = "image")]
use jolt_riscv::{JoltInstructionProfile, RV64IMAC_JOLT};

#[cfg(feature = "field-inline")]
pub use crate::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
pub use backend::{ChunkedExecutionBackend, ExecutionBackend, ExecutionSummary};
pub use error::TraceError;
#[cfg(feature = "field-inline")]
pub use trace::FieldEvent;
pub use trace::{
    ExecutionDimensions, JoltProgram, MemoryImage, RamAddressBounds, TraceData, TraceInputs,
    TraceOutput,
};

#[cfg(feature = "image")]
pub fn build_jolt_program(elf_bytes: &[u8]) -> Result<JoltProgram, ProgramError> {
    build_jolt_program_with_inline_provider(
        elf_bytes,
        &mut NoInlineExpansionProvider,
        RV64IMAC_JOLT,
    )
}

#[cfg(feature = "image")]
pub fn build_jolt_program_with_inline_provider<P: InlineExpansionProvider + ?Sized>(
    elf_bytes: &[u8],
    inline_provider: &mut P,
    profile: JoltInstructionProfile,
) -> Result<JoltProgram, ProgramError> {
    let image = decode_elf(elf_bytes, profile)?;
    let expanded_bytecode = image.expanded_bytecode(inline_provider, profile)?;
    Ok(JoltProgram::from_rv64_image_with_profile(
        elf_bytes.to_vec(),
        expanded_bytecode,
        image,
        profile,
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
