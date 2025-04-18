use serde::{Deserialize, Serialize};

pub trait InstructionFormat {
    type RegisterState: Default;

    fn parse(word: u32) -> Self;
    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, registers: [i64; 64]) {}
    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, registers: [i64; 64]) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatB {
    pub rs1: usize,
    pub rs2: usize,
    pub imm: i64,
}

#[derive(Default)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatCSR {
    pub csr: u16,
    pub rs: usize,
    pub rd: usize,
}

#[derive(Default)]
pub struct RegisterStateFormatCSR {
    rs: u64,
    rd: (u64, u64), // (old_value, new_value)
}

impl InstructionFormat for FormatCSR {
    type RegisterState = RegisterStateFormatCSR;

    fn parse(word: u32) -> Self {
        FormatCSR {
            csr: ((word >> 20) & 0xfff) as u16, // [31:20]
            rs: ((word >> 15) & 0x1f) as usize, // [19:15], also uimm
            rd: ((word >> 7) & 0x1f) as usize,  // [11:7]
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatI {
    pub rd: usize,
    pub rs1: usize,
    pub imm: i64,
}

#[derive(Default)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatJ {
    pub rd: usize,
    pub imm: i64,
}

#[derive(Default)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatR {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
}

#[derive(Default)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatR2 {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub rs3: usize,
}

#[derive(Default)]
pub struct RegisterStateFormatR2 {
    rd: (u64, u64), // (old_value, new_value)
    rs1: u64,
    rs2: u64,
    rs3: u64,
}

impl InstructionFormat for FormatR2 {
    type RegisterState = RegisterStateFormatR2;

    fn parse(word: u32) -> Self {
        FormatR2 {
            rd: ((word >> 7) & 0x1f) as usize,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as usize, // [19:15]
            rs2: ((word >> 20) & 0x1f) as usize, // [24:20]
            rs3: ((word >> 27) & 0x1f) as usize, // [31:27]
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatS {
    pub rs1: usize,
    pub rs2: usize,
    pub imm: i64,
}

#[derive(Default)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatU {
    pub rd: usize,
    pub imm: i64,
}

#[derive(Default)]
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
}
