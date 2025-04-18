use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::{
    format::{FormatR, InstructionFormat},
    RISCVCycle, RISCVInstruction, RV32IMCycle,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DIV<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatR,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for DIV<WORD_SIZE> {
    const MASK: u32 = 0xfe00707f;
    const MATCH: u32 = 0x02004033;

    type Format = FormatR;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64) -> Self {
        Self {
            address,
            operands: FormatR::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1];
        let divisor = cpu.x[self.operands.rs2];
        if divisor == 0 {
            cpu.x[self.operands.rd] = -1;
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.x[self.operands.rd] = dividend;
        } else {
            cpu.x[self.operands.rd] = cpu.sign_extend(dividend.wrapping_div(divisor))
        }
    }
}

// impl<const WORD_SIZE: usize> VirtualInstructionSequence for DIV<WORD_SIZE> {
//     fn virtual_trace(cycle: RISCVCycle<Self>) -> Vec<RV32IMCycle> {
//         todo!()
//     }

//     fn sequence_output(x: u64, y: u64) -> u64 {
//         let x = x as i32;
//         let y = y as i32;
//         if y == 0 {
//             return (1 << WORD_SIZE) - 1;
//         }
//         let mut quotient = x / y;
//         let remainder = x % y;
//         if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
//             quotient -= 1;
//         }
//         quotient as u32 as u64
//     }
// }
