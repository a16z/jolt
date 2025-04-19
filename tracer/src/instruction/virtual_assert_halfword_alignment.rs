use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::{
    format::{normalize_register_value, InstructionFormat},
    RISCVInstruction,
};

/// `VirtualAssertHalfwordAlignment` is the only instruction that
/// uses `rs1` and `imm` but does not write to a destination register.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HalfwordAlignFormat {
    pub rs1: usize,
    pub imm: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct HalfwordAlignRegisterState {
    rs1: u64,
}

impl InstructionFormat for HalfwordAlignFormat {
    type RegisterState = HalfwordAlignRegisterState;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, _: &mut Self::RegisterState, _: &mut Cpu) {
        // No register write
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VirtualAssertHalfwordAlignment<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: HalfwordAlignFormat,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for VirtualAssertHalfwordAlignment<WORD_SIZE> {
    const MASK: u32 = 0; // Virtual
    const MATCH: u32 = 0; // Virtual

    type Format = HalfwordAlignFormat;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(_: u32, _: u64) -> Self {
        unimplemented!("virtual instruction")
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let address = cpu.x[self.operands.rs1] + self.operands.imm;
        assert!(
            address & 1 == 0,
            "RAM access (LH or LHU) is not halfword aligned: {address:x}"
        );
    }
}
