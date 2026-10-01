use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_b::FormatB, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateB {
    pub rs1: u64,
    pub rs2: u64,
}

impl InstructionRegisterState for RegisterStateB {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1.unwrap() == 0 {
            0
        } else {
            rng.next_u64()
        };

        Self {
            rs1: rs1_value,
            rs2: if operands.rs2.unwrap() == 0 {
                0
            } else if operands.rs2 == operands.rs1 {
                rs1_value
            } else {
                rng.next_u64()
            },
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rs2_value(&self) -> Option<u64> {
        Some(self.rs2)
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateB
where
    I: RISCVInstruction<Format = FormatB>,
{
    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self {
        let operands = instruction.operands();
        Self {
            rs1: normalize_register_value(cpu, operands.rs1 as usize),
            rs2: normalize_register_value(cpu, operands.rs2 as usize),
        }
    }

    fn capture_post(&mut self, _instruction: &I, _cpu: &Cpu) {}
}
