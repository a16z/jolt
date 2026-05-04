use jolt_riscv::NormalizedInstruction;

use crate::ProgramError;

pub fn decode_instruction(
    _word: u32,
    _address: u64,
) -> Result<NormalizedInstruction, ProgramError> {
    Err(ProgramError::UnsupportedArchitecture(
        "RV64 instruction decoder is not wired yet",
    ))
}
