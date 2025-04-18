use serde::{Deserialize, Serialize};

use super::{
    format::{FormatB, InstructionFormat},
    RISCVInstruction,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BNE<const WORD_SIZE: usize> {
    pub address: u64,
    pub instruction: FormatB,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for BNE<WORD_SIZE> {
    type Format = FormatB;
    type RAMAccess = ();

    fn to_raw(&self) -> Self::Format {
        self.instruction
    }

    fn new(word: u32, address: u64) -> Self {
        Self {
            address,
            instruction: FormatB::parse(word),
            virtual_sequence_remaining: None,
        }
    }
}
