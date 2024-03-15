use serde::{Deserialize, Serialize};
use strum_macros::FromRepr;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct RVTraceRow {
    pub instruction: ELFInstruction,
    pub register_state: RegisterState,
    pub memory_state: Option<MemoryState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ELFInstruction {
    pub address: u64,
    pub opcode: RV32IM,
    pub raw: u32,
    pub rs1: Option<u64>,
    pub rs2: Option<u64>,
    pub rd: Option<u64>,
    pub imm: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RegisterState {
    pub rs1_val: Option<u64>,
    pub rs2_val: Option<u64>,
    pub rd_pre_val: Option<u64>,
    pub rd_post_val: Option<u64>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum MemoryState {
    Read {
        address: u64,
        value: u64,
    },
    Write {
        address: u64,
        pre_value: u64,
        post_value: u64,
    },
}

impl RVTraceRow {
    pub fn imm_u64(&self) -> u64 {
        match self.instruction.opcode.instruction_type() {
            RV32InstructionFormat::R => unimplemented!("R type does not use imm u64"),
            RV32InstructionFormat::I => self.instruction.imm.unwrap() as u64,
            RV32InstructionFormat::U => ((self.instruction.imm.unwrap() as u32) << 12u32) as u64,
            RV32InstructionFormat::S => unimplemented!("S type does not use imm u64"),
            // UJ-type instructions point to address offsets: even numbers.
            // TODO(JOLT-88): De-normalizing was already done elsewhere. Should make this is consistent.
            RV32InstructionFormat::UJ => (self.instruction.imm.unwrap() as u64) << 0u64,
            _ => unimplemented!(),
        }
    }
}

// Reference: https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf
#[derive(Debug, PartialEq, Eq, Clone, Copy, FromRepr, Serialize, Deserialize)]
#[repr(u8)]
pub enum RV32IM {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
    SLL,
    SRL,
    SRA,
    SLT,
    SLTU,
    ADDI,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
    SLTI,
    SLTIU,
    LB,
    LH,
    LW,
    LBU,
    LHU,
    SB,
    SH,
    SW,
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
    JAL,
    JALR,
    LUI,
    AUIPC,
    ECALL,
    EBREAK,
    MUL,
    MULH,
    MULSU,
    MULU,
    DIV,
    DIVU,
    REM,
    REMU,
    UNIMPL,
}

impl RV32IM {
    pub fn from_str(s: &str) -> Self {
        match s {
            "ADD" => Self::ADD,
            "SUB" => Self::SUB,
            "XOR" => Self::XOR,
            "OR" => Self::OR,
            "AND" => Self::AND,
            "SLL" => Self::SLL,
            "SRL" => Self::SRL,
            "SRA" => Self::SRA,
            "SLT" => Self::SLT,
            "SLTU" => Self::SLTU,
            "ADDI" => Self::ADDI,
            "XORI" => Self::XORI,
            "ORI" => Self::ORI,
            "ANDI" => Self::ANDI,
            "SLLI" => Self::SLLI,
            "SRLI" => Self::SRLI,
            "SRAI" => Self::SRAI,
            "SLTI" => Self::SLTI,
            "SLTIU" => Self::SLTIU,
            "LB" => Self::LB,
            "LH" => Self::LH,
            "LW" => Self::LW,
            "LBU" => Self::LBU,
            "LHU" => Self::LHU,
            "SB" => Self::SB,
            "SH" => Self::SH,
            "SW" => Self::SW,
            "BEQ" => Self::BEQ,
            "BNE" => Self::BNE,
            "BLT" => Self::BLT,
            "BGE" => Self::BGE,
            "BLTU" => Self::BLTU,
            "BGEU" => Self::BGEU,
            "JAL" => Self::JAL,
            "JALR" => Self::JALR,
            "LUI" => Self::LUI,
            "AUIPC" => Self::AUIPC,
            "ECALL" => Self::ECALL,
            "EBREAK" => Self::EBREAK,
            "MUL" => Self::MUL,
            "MULH" => Self::MULH,
            "MULSU" => Self::MULSU,
            "MULU" => Self::MULU,
            "DIV" => Self::DIV,
            "DIVU" => Self::DIVU,
            "REM" => Self::REM,
            "REMU" => Self::REMU,
            "UNIMPL" => Self::UNIMPL,
            _ => panic!("Could not match instruction to RV32IM set."),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum RV32InstructionFormat {
    R,
    I,
    S,
    SB,
    U,
    UJ,
}

impl RV32IM {
    #[rustfmt::skip] // keep matches pretty
    pub fn instruction_type(&self) -> RV32InstructionFormat {
        match self {
            RV32IM::ADD
            | RV32IM::SUB
            | RV32IM::XOR
            | RV32IM::OR
            | RV32IM::AND
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::MUL
            | RV32IM::MULH
            | RV32IM::MULSU
            | RV32IM::MULU
            | RV32IM::DIV
            | RV32IM::DIVU
            | RV32IM::REM
            | RV32IM::REMU => RV32InstructionFormat::R,

            RV32IM::ADDI
            | RV32IM::XORI
            | RV32IM::ORI
            | RV32IM::ANDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLTI
            | RV32IM::SLTIU => RV32InstructionFormat::I,

            RV32IM::LB 
            | RV32IM::LH 
            | RV32IM::LW 
            | RV32IM::LBU 
            | RV32IM::LHU 
            | RV32IM::JALR => {
                RV32InstructionFormat::I
            }

            RV32IM::SB 
            | RV32IM::SH 
            | RV32IM::SW => RV32InstructionFormat::S,

            RV32IM::BEQ 
            | RV32IM::BNE 
            | RV32IM::BLT 
            | RV32IM::BGE 
            | RV32IM::BLTU 
            | RV32IM::BGEU => {
                RV32InstructionFormat::SB
            }

            RV32IM::LUI 
            | RV32IM::AUIPC => RV32InstructionFormat::U,

            RV32IM::JAL => RV32InstructionFormat::UJ,

            RV32IM::ECALL 
            | RV32IM::EBREAK 
            | RV32IM::UNIMPL => unimplemented!(),
        }
    }
}

pub fn to_ram_address(index: usize) -> usize {
    index * constants::BYTES_PER_INSTRUCTION + constants::RAM_START_ADDRESS as usize
}

pub mod constants;
pub mod field_conversion;
pub mod path;
pub mod serializable;
pub mod parallel;