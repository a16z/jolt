use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualRightShiftWI {
    pub rd: u8,
    pub rs1: u8,
    pub imm: u64,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateVirtualRightShiftWI {
    pub rd: (u64, u64),
    pub rs1: u64,
}

impl Default for RegisterStateVirtualRightShiftWI {
    fn default() -> Self {
        Self { rd: (0, 0), rs1: 1 }
    }
}

impl InstructionRegisterState for RegisterStateVirtualRightShiftWI {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;

        let rs1 = if operands.rs1 == Some(0) {
            0
        } else {
            rng.next_u64()
        };
        Self {
            rd: (
                if operands.rd == operands.rs1 {
                    rs1
                } else {
                    rng.next_u64()
                },
                rng.next_u64(),
            ),
            rs1,
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl InstructionFormat for FormatVirtualRightShiftWI {
    type RegisterState = RegisterStateVirtualRightShiftWI;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu, self.rs1 as usize);
        state.rd.0 = normalize_register_value(cpu, self.rd as usize);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu, self.rd as usize);
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;

        let shift = rng.next_u64() & 0x1f;
        Self {
            imm: (1u64 << 32) - (1u64 << shift),
            rd: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
            rs1: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatVirtualRightShiftWI {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatVirtualRightShiftWI> for NormalizedOperands {
    fn from(format: FormatVirtualRightShiftWI) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: None,
            imm: format.imm as i128,
        }
    }
}
