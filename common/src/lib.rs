use strum_macros::FromRepr;

#[derive(Debug, PartialEq)]
pub struct RVTraceRow {
    pub instruction: Instruction,
    pub register_state: RegisterState,
    pub memory_state: Option<MemoryState>,
}

#[derive(Debug, PartialEq)]
pub struct Instruction {
    pub address: u64,
    pub opcode: &'static str,
    pub rs1: Option<u64>,
    pub rs2: Option<u64>,
    pub rd: Option<u64>,
    pub imm: Option<i32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RegisterState {
    pub rs1_val: Option<u64>,
    pub rs2_val: Option<u64>,
    pub rd_pre_val: Option<u64>,
    pub rd_post_val: Option<u64>,
}

#[derive(Debug, PartialEq)]
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

// Reference: https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf
#[derive(Debug, PartialEq, Eq, Clone, Copy, FromRepr)]
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
        RV32IM::ADD | RV32IM::SUB | RV32IM::XOR | RV32IM::OR | RV32IM::AND
        | RV32IM::SLL | RV32IM::SRL | RV32IM::SRA | RV32IM::SLT | RV32IM::SLTU
        | RV32IM::MUL | RV32IM::MULH | RV32IM::MULSU | RV32IM::MULU
        | RV32IM::DIV | RV32IM::DIVU | RV32IM::REM | RV32IM::REMU => RV32InstructionFormat::R,
        
        RV32IM::ADDI | RV32IM::XORI | RV32IM::ORI | RV32IM::ANDI
        | RV32IM::SLLI | RV32IM::SRLI | RV32IM::SRAI | RV32IM::SLTI | RV32IM::SLTIU => RV32InstructionFormat::I,
        
        RV32IM::LB | RV32IM::LH | RV32IM::LW | RV32IM::LBU | RV32IM::LHU
        | RV32IM::SB | RV32IM::SH | RV32IM::SW => RV32InstructionFormat::S,
        
        RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU => RV32InstructionFormat::SB,
        
        RV32IM::LUI | RV32IM::AUIPC => RV32InstructionFormat::U,
        
        RV32IM::JAL | RV32IM::JALR => RV32InstructionFormat::UJ,
        
        RV32IM::ECALL | RV32IM::EBREAK => unimplemented!(),
      }
    }
  }

pub mod constants;