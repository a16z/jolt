use crate::emulator::cpu::Cpu;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::format_r::RegisterStateFormatR;
use super::{normalize_register_value, InstructionFormat, NormalizedOperands};

/// `rd = rotr(rs1 ^ rs2, rotation)`; the rotation rides in the row's
/// immediate and selects the lookup table.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualXorRot {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
    pub rotation: u32,
}

impl InstructionFormat for FormatVirtualXorRot {
    type RegisterState = RegisterStateFormatR;

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
    fn random(rng: &mut StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use jolt_riscv::instructions::XOR_ROT_ROTATIONS;
        use rand::RngCore;
        Self {
            rd: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
            rs1: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
            rs2: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
            rotation: XOR_ROT_ROTATIONS[rng.next_u64() as usize % XOR_ROT_ROTATIONS.len()],
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatVirtualXorRot {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
            rotation: operands.imm as u32,
        }
    }
}

impl From<FormatVirtualXorRot> for NormalizedOperands {
    fn from(format: FormatVirtualXorRot) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: format.rotation as i128,
        }
    }
}
