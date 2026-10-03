use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_t::FormatT, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateT {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStateT {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1.unwrap() == 0 {
            0
        } else {
            rng.next_u64()
        };

        Self {
            rd: (
                match operands.rd {
                    _ if operands.rd == operands.rs1 => rs1_value,
                    _ => rng.next_u64(),
                },
                rng.next_u64(),
            ),
            rs1: rs1_value,
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateT
where
    I: RISCVInstruction<Format = FormatT>,
{
    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self {
        let operands = instruction.operands();
        let rd = normalize_register_value(cpu, operands.rd as usize);
        Self {
            rd: (rd, rd),
            rs1: normalize_register_value(cpu, operands.rs1 as usize),
        }
    }

    fn capture_post(&mut self, instruction: &I, cpu: &Cpu) {
        self.rd.1 = normalize_register_value(cpu, instruction.operands().rd as usize);
    }
}
