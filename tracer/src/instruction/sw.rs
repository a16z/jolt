use serde::{Deserialize, Serialize};

use super::MemoryWrite;

use super::{
    format::{FormatS, InstructionFormat},
    RISCVInstruction,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SW<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatS,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for SW<WORD_SIZE> {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x00002023;

    type Format = FormatS;
    type RAMAccess = MemoryWrite;

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64) -> Self {
        Self {
            address,
            operands: FormatS::parse(word),
            virtual_sequence_remaining: None,
        }
    }
}
