use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualRightShiftR<const MASK_WIDTH: usize = 64> {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
}

impl<const MASK_WIDTH: usize> Default for FormatVirtualRightShiftR<MASK_WIDTH> {
    fn default() -> Self {
        Self {
            rd: 0,
            rs1: 1,
            rs2: 2,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateVirtualRightShift<const MASK_WIDTH: usize = 64> {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
    pub rs2: u64,
}

impl<const MASK_WIDTH: usize> InstructionRegisterState
    for RegisterStateVirtualRightShift<MASK_WIDTH>
{
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1.unwrap() == 0 {
            0
        } else {
            rng.next_u64()
        };

        assert!((1..=64).contains(&MASK_WIDTH));
        let shift = rng.next_u64() % MASK_WIDTH as u64;

        debug_assert_ne!(
            operands.rs2.unwrap(),
            0,
            "rs2 cannot be 0 in VirtualRightShift instruction"
        );
        debug_assert_ne!(
            operands.rs2, operands.rs1,
            "rs2 cannot equal rs1 in VirtualRightShift instruction"
        );

        let rs2_value = ((1u128 << MASK_WIDTH) - (1u128 << shift)) as u64;

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

impl<const MASK_WIDTH: usize> InstructionFormat for FormatVirtualRightShiftR<MASK_WIDTH> {
    type RegisterState = RegisterStateVirtualRightShift<MASK_WIDTH>;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu, self.rs1 as usize);
        state.rs2 = normalize_register_value(cpu, self.rs2 as usize);
        state.rd.0 = normalize_register_value(cpu, self.rd as usize);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu, self.rd as usize);
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

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl<const MASK_WIDTH: usize> From<NormalizedOperands> for FormatVirtualRightShiftR<MASK_WIDTH> {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
        }
    }
}

impl<const MASK_WIDTH: usize> From<FormatVirtualRightShiftR<MASK_WIDTH>> for NormalizedOperands {
    fn from(format: FormatVirtualRightShiftR<MASK_WIDTH>) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: 0,
        }
    }
}
