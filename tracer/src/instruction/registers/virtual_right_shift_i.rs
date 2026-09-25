use crate::{
    emulator::cpu::Cpu,
    instruction::{
        format::format_virtual_right_shift_i::FormatVirtualRightShiftI, RISCVInstruction,
    },
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateVirtualRightShiftI {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
}

impl Default for RegisterStateVirtualRightShiftI {
    fn default() -> Self {
        Self { rd: (0, 0), rs1: 1 }
    }
}

impl InstructionRegisterState for RegisterStateVirtualRightShiftI {
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
                if operands.rd == operands.rs1 {
                    rs1_value
                } else {
                    rng.next_u64()
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

impl<I, const MASK_WIDTH: usize> RegisterSnapshot<I> for RegisterStateVirtualRightShiftI
where
    I: RISCVInstruction<Format = FormatVirtualRightShiftI<MASK_WIDTH>>,
{
    type Before = (u64, u64);

    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self::Before {
        let operands = instruction.operands();
        (
            normalize_register_value(cpu, operands.rs1 as usize),
            normalize_register_value(cpu, operands.rd as usize),
        )
    }

    fn capture_post(instruction: &I, before: Self::Before, cpu: &Cpu) -> Self {
        let (rs1, rd_pre) = before;
        Self {
            rd: (
                rd_pre,
                normalize_register_value(cpu, instruction.operands().rd as usize),
            ),
            rs1,
        }
    }
}
