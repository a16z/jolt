use ark_ff::PrimeField;
use strum_macros::FromRepr;

use crate::jolt::vm::{pc::ELFRow, test_vm::TestInstructionSet};

use super::{JoltProvableTrace, MemoryOp};

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

struct RVTraceRow {
  pc: u64,
  opcode: RV32IM,

  rd: Option<u64>,
  rs1: Option<u64>,
  rs2: Option<u64>,
  imm: Option<u64>,

  rd_pre_val: Option<u64>,
  rd_post_val: Option<u64>,

  rs1_val: Option<u64>,
  rs2_val: Option<u64>,

  memory_bytes_before: Option<Vec<u8>>,
  memory_bytes_after: Option<Vec<u8>>,
}

impl JoltProvableTrace for RVTraceRow {
  type JoltInstructionEnum = TestInstructionSet;
  fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum> {
    // Handle fan-out 1-to-many
    todo!("massive match");
    // vec![TestInstructionSet::from(self.opcode)]
  }

  fn to_ram_ops(&self) -> Vec<MemoryOp> {
    todo!("massive match")
  }

  fn to_pc_trace(&self) -> ELFRow {
    // TODO(sragss): Is 0 padding correct?
    ELFRow::new(
      self.pc.try_into().unwrap(),
      self.opcode as u64,
      self.rd.unwrap_or(0),
      self.rs1.unwrap_or(0),
      self.rs2.unwrap_or(0),
      self.imm.unwrap_or(0),
    )
  }

  fn to_circuit_flags<F: PrimeField>(&self) -> Vec<F> {
    // Jolt Appendix A.1
    // 0: first_operand == rs1 (1 if PC)
    // 1: second_operand == rs2 (1 if imm)
    // 2: Load instruction
    // 3: Store instruciton
    // 4: Jump instruction
    // 5: Branch instruciton
    // 6: Instruction writes lookup output to rd
    // 7: Instruction adds operands (ie, and uses the ADD lookup table)
    // 8: Instruction subtracts operands
    // 9: Instruction multiplies operands
    // 10: Instruction involves non-deterministic advice?
    // 11: Instruction asserts lookup output as false
    // 12: Instruction asserts lookup output as true
    // 13: Sign-bit of imm
    // 14: Instruction is lui

    let flag_0 = match self.opcode {
      RV32IM::JAL | RV32IM::JALR | RV32IM::LUI | RV32IM::AUIPC => true,
      _ => false,
    };

    let flag_1 = match self.opcode {
      RV32IM::ADDI
      | RV32IM::XORI
      | RV32IM::ORI
      | RV32IM::ANDI
      | RV32IM::SLLI
      | RV32IM::SRLI
      | RV32IM::SRAI
      | RV32IM::SLTI
      | RV32IM::SLTIU => true,
      _ => false,
    };

    let flag_2 = match self.opcode {
      RV32IM::LB | RV32IM::LH | RV32IM::LW | RV32IM::LBU | RV32IM::LHU => true,
      _ => false,
    };

    let flag_3 = match self.opcode {
      RV32IM::SB | RV32IM::SH | RV32IM::SW => true,
      _ => false,
    };

    let flag_4 = match self.opcode {
      RV32IM::JAL | RV32IM::JALR => true,
      _ => false,
    };

    let flag_5 = match self.opcode {
      RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU => true,
      _ => false,
    };

    // loads, stores, branches, jumps do not store the lookup output to rd (they may update rd in other ways)
    let flag_6 = match self.opcode {
      RV32IM::LB
      | RV32IM::LH
      | RV32IM::LW
      | RV32IM::LBU
      | RV32IM::LHU
      | RV32IM::SB
      | RV32IM::SH
      | RV32IM::SW
      | RV32IM::BEQ
      | RV32IM::BNE
      | RV32IM::BLT
      | RV32IM::BGE
      | RV32IM::BLTU
      | RV32IM::BGEU
      | RV32IM::JAL
      | RV32IM::JALR
      | RV32IM::LUI => false,
      _ => true,
    };

    let flag_7 = match self.opcode {
      RV32IM::ADD | RV32IM::ADDI | RV32IM::JAL | RV32IM::JALR | RV32IM::AUIPC => true,
      _ => false,
    };

    let flag_8 = match self.opcode {
      RV32IM::SUB => true,
      _ => false,
    };

    let flag_9 = match self.opcode {
      RV32IM::MUL | RV32IM::MULU | RV32IM::MULH | RV32IM::MULSU => true,
      _ => false,
    };

    // not incorporating advice instructions yet
    let flag_10 = match self.opcode {
      _ => false,
    };

    // not incorporating assert true instructions yet
    let flag_11 = match self.opcode {
      _ => false,
    };

    // not incorporating assert false instructions yet
    let flag_12 = match self.opcode {
      _ => false,
    };

    // not incorporating advice instructions yet
    let flag_13 = match self.opcode {
      _ => false,
    };

    let flag_14 = match self.opcode {
      RV32IM::LUI => true,
      _ => false,
    };

    vec![
      F::from(flag_0),
      F::from(flag_1),
      F::from(flag_2),
      F::from(flag_3),
      F::from(flag_4),
      F::from(flag_5),
      F::from(flag_6),
      F::from(flag_7),
      F::from(flag_8),
      F::from(flag_9),
      F::from(flag_10),
      F::from(flag_11),
      F::from(flag_12),
      F::from(flag_13),
      F::from(flag_13),
      F::from(flag_14),
    ]
  }
}
