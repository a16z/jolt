use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_u::FormatU, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateU {
    pub rd: (u64, u64), // (old_value, new_value)
}

impl InstructionRegisterState for RegisterStateU {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, _operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        Self {
            rd: (rng.next_u64(), rng.next_u64()),
        }
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateU
where
    I: RISCVInstruction<Format = FormatU>,
{
    type Before = u64;

    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self::Before {
        let operands = instruction.operands();
        normalize_register_value(cpu, operands.rd as usize)
    }

    fn capture_post(instruction: &I, before: Self::Before, cpu: &Cpu) -> Self {
        let rd_pre = before;
        Self {
            rd: (
                rd_pre,
                normalize_register_value(cpu, instruction.operands().rd as usize),
            ),
        }
    }
}
