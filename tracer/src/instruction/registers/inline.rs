use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_inline::FormatInline, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateInline {
    pub rs1: u64,
    pub rs2: u64,
    pub rs3: u64,
}

impl InstructionRegisterState for RegisterStateInline {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1.unwrap() == 0 {
            0
        } else {
            rng.next_u64()
        };

        let rs2_value = match operands.rs2.unwrap() {
            0 => 0,
            _ if operands.rs2 == operands.rs1 => rs1_value,
            _ => rng.next_u64(),
        };

        // Note: operands.rd maps to rs3 in FormatInline (see From implementations)
        let rs3_value = match operands.rd {
            _ if operands.rd == operands.rs1 => rs1_value,
            _ if operands.rd == operands.rs2 => rs2_value,
            _ => rng.next_u64(),
        };

        Self {
            rs1: rs1_value,
            rs2: rs2_value,
            rs3: rs3_value,
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rs2_value(&self) -> Option<u64> {
        Some(self.rs2)
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateInline
where
    I: RISCVInstruction<Format = FormatInline>,
{
    type Before = Self;

    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self::Before {
        let operands = instruction.operands();
        Self {
            rs1: normalize_register_value(cpu, operands.rs1 as usize),
            rs2: normalize_register_value(cpu, operands.rs2 as usize),
            rs3: normalize_register_value(cpu, operands.rs3 as usize),
        }
    }

    fn capture_post(_instruction: &I, before: Self::Before, _cpu: &Cpu) -> Self {
        before
    }
}
