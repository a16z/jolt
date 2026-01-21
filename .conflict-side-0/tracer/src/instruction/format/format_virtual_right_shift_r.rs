use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualRightShiftR {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
}

impl Default for FormatVirtualRightShiftR {
    fn default() -> Self {
        Self {
            rd: 0,
            rs1: 1,
            rs2: 2,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateVirtualRightShift {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
    pub rs2: u64,
}

impl InstructionRegisterState for RegisterStateVirtualRightShift {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1.unwrap() == 0 {
            0
        } else {
            rng.next_u64()
        };

        let shift = rng.next_u64() & 0x3F;
        let ones = (1u128 << (64 - shift)) - 1;

        debug_assert_ne!(
            operands.rs2.unwrap(),
            0,
            "rs2 cannot be 0 in VirtualRightShift instruction"
        );
        debug_assert_ne!(
            operands.rs2, operands.rs1,
            "rs2 cannot equal rs1 in VirtualRightShift instruction"
        );

        let rs2_value = (ones << shift) as u64;

        Self {
            rd: (
                match operands.rd {
                    _ if operands.rd == operands.rs1 => rs1_value,
                    _ if operands.rd == operands.rs2 => rs2_value,
                    _ => rng.next_u64(),
                },
                rng.next_u64(),
            ),
            rs1: rs1_value,
            rs2: rs2_value,
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rs2_value(&self) -> Option<u64> {
        Some(self.rs2)
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl InstructionFormat for FormatVirtualRightShiftR {
    type RegisterState = RegisterStateVirtualRightShift;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2 as usize], &cpu.xlen);
        state.rd.0 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        let rd = rng.next_u64() as u8 % RISCV_REGISTER_COUNT;
        let rs1 = rng.next_u64() as u8 % RISCV_REGISTER_COUNT;

        // Ensure rs2 is non-zero and different from rs1
        let mut rs2 = 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1));
        if rs2 == rs1 {
            rs2 = if rs2 == RISCV_REGISTER_COUNT - 1 {
                1
            } else {
                rs2 + 1
            };
        }

        Self { rd, rs1, rs2 }
    }
}

impl From<NormalizedOperands> for FormatVirtualRightShiftR {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
        }
    }
}

impl From<FormatVirtualRightShiftR> for NormalizedOperands {
    fn from(format: FormatVirtualRightShiftR) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: 0,
        }
    }
}
