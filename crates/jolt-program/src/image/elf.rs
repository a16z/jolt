use jolt_riscv::NormalizedInstruction;

use crate::ProgramError;

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodedProgramImage {
    pub instructions: Vec<NormalizedInstruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub program_end: u64,
    pub entry_address: u64,
}

pub fn decode_elf(_elf: &[u8]) -> Result<DecodedProgramImage, ProgramError> {
    Err(ProgramError::MalformedImage(
        "jolt-program ELF decoding is not wired yet",
    ))
}
