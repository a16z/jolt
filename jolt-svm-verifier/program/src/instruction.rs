use solana_program::program_error::ProgramError;

pub enum VerifierInstruction {
    VerifyHyperKZG,
    VerifySumcheck,
}

impl VerifierInstruction {
    pub fn unpack(input: &[u8]) -> Result<Self, ProgramError> {
        let (&tag, _rest) = input.split_first().ok_or(ProgramError::InvalidInstructionData)?;
        Ok(match tag {
            0 => VerifierInstruction::VerifyHyperKZG,
            1 => VerifierInstruction::VerifySumcheck,
            _ => return Err(ProgramError::InvalidInstructionData),
        })
    }
}