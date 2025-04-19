use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

pub trait InstructionFormat: Default {
    type RegisterState: Default + Serialize + DeserializeOwned;

    fn parse(word: u32) -> Self;
    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu);
    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu);
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatB {
    pub rs1: usize,
    pub rs2: usize,
    pub imm: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct RegisterStateFormatB {
    rs1: u64,
    rs2: u64,
}

impl InstructionFormat for FormatB {
    type RegisterState = RegisterStateFormatB;

    fn parse(word: u32) -> Self {
        FormatB {
            rs1: ((word >> 15) & 0x1f) as usize, // [19:15]
            rs2: ((word >> 20) & 0x1f) as usize, // [24:20]
            imm: (
                match word & 0x80000000 { // imm[31:12] = [31]
				0x80000000 => 0xfffff000,
				_ => 0
			} |
			((word << 4) & 0x00000800) | // imm[11] = [7]
			((word >> 20) & 0x000007e0) | // imm[10:5] = [30:25]
			((word >> 7) & 0x0000001e)
                // imm[4:1] = [11:8]
            ) as i32 as i64,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, _: &mut Self::RegisterState, _: &mut Cpu) {
        // No register write
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatI {
    pub rd: usize,
    pub rs1: usize,
    pub imm: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct RegisterStateFormatI {
    rd: (u64, u64), // (old_value, new_value)
    rs1: u64,
}

impl InstructionFormat for FormatI {
    type RegisterState = RegisterStateFormatI;

    fn parse(word: u32) -> Self {
        FormatI {
            rd: ((word >> 7) & 0x1f) as usize,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as usize, // [19:15]
            imm: (
                match word & 0x80000000 {
                    // imm[31:11] = [31]
                    0x80000000 => 0xfffff800,
                    _ => 0,
                } | ((word >> 20) & 0x000007ff)
                // imm[10:0] = [30:20]
            ) as i32 as i64,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1], &cpu.xlen);
        state.rd.0 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatJ {
    pub rd: usize,
    pub imm: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct RegisterStateFormatJ {
    rd: (u64, u64), // (old_value, new_value)
}

impl InstructionFormat for FormatJ {
    type RegisterState = RegisterStateFormatJ;

    fn parse(word: u32) -> Self {
        FormatJ {
            rd: ((word >> 7) & 0x1f) as usize, // [11:7]
            imm: (
                match word & 0x80000000 { // imm[31:20] = [31]
				0x80000000 => 0xfff00000,
				_ => 0
			} |
			(word & 0x000ff000) | // imm[19:12] = [19:12]
			((word & 0x00100000) >> 9) | // imm[11] = [20]
			((word & 0x7fe00000) >> 20)
                // imm[10:1] = [30:21]
            ) as i32 as i64,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.0 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatR {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
}

#[derive(Default, Serialize, Deserialize)]
pub struct RegisterStateFormatR {
    rd: (u64, u64), // (old_value, new_value)
    rs1: u64,
    rs2: u64,
}

impl InstructionFormat for FormatR {
    type RegisterState = RegisterStateFormatR;

    fn parse(word: u32) -> Self {
        FormatR {
            rd: ((word >> 7) & 0x1f) as usize,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as usize, // [19:15]
            rs2: ((word >> 20) & 0x1f) as usize, // [24:20]
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2], &cpu.xlen);
        state.rd.0 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatS {
    pub rs1: usize,
    pub rs2: usize,
    pub imm: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct RegisterStateFormatS {
    rs1: u64,
    rs2: u64,
}

impl InstructionFormat for FormatS {
    type RegisterState = RegisterStateFormatS;

    fn parse(word: u32) -> Self {
        FormatS {
            rs1: ((word >> 15) & 0x1f) as usize, // [19:15]
            rs2: ((word >> 20) & 0x1f) as usize, // [24:20]
            imm: (
                match word & 0x80000000 {
				0x80000000 => 0xfffff000,
				_ => 0
			} | // imm[31:12] = [31]
			((word >> 20) & 0xfe0) | // imm[11:5] = [31:25]
			((word >> 7) & 0x1f)
                // imm[4:0] = [11:7]
            ) as i32 as i64,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, _: &mut Self::RegisterState, _: &mut Cpu) {
        // No register write
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatU {
    pub rd: usize,
    pub imm: i64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct RegisterStateFormatU {
    rd: (u64, u64), // (old_value, new_value)
}

impl InstructionFormat for FormatU {
    type RegisterState = RegisterStateFormatU;

    fn parse(word: u32) -> Self {
        FormatU {
            rd: ((word >> 7) & 0x1f) as usize, // [11:7]
            imm: (
                match word & 0x80000000 {
				0x80000000 => 0xffffffff00000000,
				_ => 0
			} | // imm[63:32] = [31]
			((word as u64) & 0xfffff000)
                // imm[31:12] = [31:12]
            ) as i32 as i64,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.0 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }
}

pub fn normalize_register_value(value: i64, xlen: &Xlen) -> u64 {
    match xlen {
        Xlen::Bit32 => value as u32 as u64,
        Xlen::Bit64 => value as u64,
    }
}
