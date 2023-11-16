use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use strum_macros::{EnumCount, EnumIter, FromRepr};

use crate::jolt::instruction::and::ANDInstruction;
use crate::jolt::instruction::bltu::BLTUInstruction;
use crate::jolt::instruction::or::ORInstruction;
use crate::jolt::instruction::sll::SLLInstruction;
use crate::jolt::instruction::slt::SLTInstruction;
use crate::jolt::instruction::xor::XORInstruction;
use crate::jolt::instruction::{add::ADDInstruction, sub::SUBInstruction};
use crate::jolt::instruction::{JoltInstruction, Opcode};
use crate::jolt::subtable::LassoSubtable;
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
  fn instruction_type(&self) -> RV32InstructionFormat {
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
      | RV32IM::SB
      | RV32IM::SH
      | RV32IM::SW => RV32InstructionFormat::S,
      RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU => {
        RV32InstructionFormat::SB
      }
      RV32IM::LUI | RV32IM::AUIPC => RV32InstructionFormat::U,
      RV32IM::JAL | RV32IM::JALR => RV32InstructionFormat::UJ,
      RV32IM::ECALL | RV32IM::EBREAK => unimplemented!(),
    }
  }
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

impl RVTraceRow {
  fn new(
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
  ) -> Self {
    let res = Self {
      pc,
      opcode,
      rd,
      rs1,
      rs2,
      imm,
      rd_pre_val,
      rd_post_val,
      rs1_val,
      rs2_val,
      memory_bytes_before,
      memory_bytes_after,
    };
    res.validate();
    res
  }

  fn RType(
    pc: u64,
    opcode: RV32IM,
    rd: u64,
    rs1: u64,
    rs2: u64,
    rd_pre_val: u64,
    rd_post_val: u64,
    rs1_val: u64,
    rs2_val: u64,
  ) -> Self {
    assert_eq!(opcode.instruction_type(), RV32InstructionFormat::R);
    Self::new(
      pc,
      opcode,
      Some(rd),
      Some(rs1),
      Some(rs2),
      None,
      Some(rd_pre_val),
      Some(rd_post_val),
      Some(rs1_val),
      Some(rs2_val),
      None,
      None,
    )
  }

  fn UType(pc: u64, opcode: RV32IM, rd: u64, rd_pre_val: u64, rd_post_val: u64, imm: u64) -> Self {
    assert_eq!(opcode.instruction_type(), RV32InstructionFormat::U);
    Self::new(
      pc,
      opcode,
      Some(rd),
      None,
      None,
      Some(imm),
      Some(rd_pre_val),
      Some(rd_post_val),
      None,
      None,
      None,
      None,
    )
  }

  fn validate(&self) {
    // TODO(sragss): Assert sizing of values depending on instruction format.
    // TODO(sragss): Assert register addresses are in our preconfigured region.
    match self.opcode.instruction_type() {
      RV32InstructionFormat::R => {
        assert!(self.rd.is_some());
        assert!(self.rd_pre_val.is_some());
        assert!(self.rd_post_val.is_some());
        assert!(self.rs1.is_some());
        assert!(self.rs1_val.is_some());
        assert!(self.rs2.is_some());
        assert!(self.rs2_val.is_some());
        assert!(self.imm.is_none());

        assert!(self.memory_bytes_before.is_none());
        assert!(self.memory_bytes_after.is_none());
      }
      RV32InstructionFormat::I => todo!(),
      RV32InstructionFormat::S => todo!(),
      RV32InstructionFormat::SB => todo!(),
      RV32InstructionFormat::U => {
        assert!(self.rd.is_some());
        assert!(self.rd_pre_val.is_some());
        assert!(self.rd_post_val.is_some());
        assert!(self.imm.is_some());
        assert_eq!(self.rd_post_val.unwrap(), self.imm.unwrap());

        assert!(self.rs1.is_none());
        assert!(self.rs1_val.is_none());
        assert!(self.rs2.is_none());
        assert!(self.rs2_val.is_none());

        assert!(self.memory_bytes_before.is_none());
        assert!(self.memory_bytes_after.is_none());

      }
      RV32InstructionFormat::UJ => todo!(),
    }

      // Check memory_before / memory_after
      let assert_load = |size: usize| {
        assert!(self.memory_bytes_before.is_some());
        assert!(self.memory_bytes_after.is_none());
        assert_eq!(self.memory_bytes_before.as_ref().unwrap().len(), size);
      };
      let assert_store= |size: usize| {
        assert!(self.memory_bytes_before.is_some());
        assert!(self.memory_bytes_after.is_some());
        assert_eq!(self.memory_bytes_before.as_ref().unwrap().len(), size);
        assert_eq!(self.memory_bytes_after.as_ref().unwrap().len(), size);
      };
      match self.opcode {
        RV32IM::LB | RV32IM::LBU => assert_load(1),
        RV32IM::LH | RV32IM::LHU => assert_load(2),
        RV32IM::LW => assert_load(4),
        RV32IM::SB => assert_store(1),
        RV32IM::SH => assert_store(2),
        RV32IM::SW => assert_store(4),
        _ => {}
      };
  }
}

#[test]
fn test() {
  let trace = vec![
    RVTraceRow::UType(0, RV32IM::LUI, 10, 0, 15, 15),
    RVTraceRow::UType(1, RV32IM::LUI, 11, 0, 20, 20),
    RVTraceRow::RType(2, RV32IM::ADD, 12, 10, 11, 0, 35, 15, 20),
  ];
}

#[repr(u8)]
#[derive(Copy, Clone, EnumIter, EnumCount)]
#[enum_dispatch(JoltInstruction)]
pub enum RVLookups {
  ADD(ADDInstruction),
  SUB(SUBInstruction),
  XOR(XORInstruction),
  OR(ORInstruction),
  AND(ANDInstruction),
  SLL(SLLInstruction),
  // SRL(SRLInstruction),
  // SRA(),
  SLT(SLTInstruction),
  // SLTU(),
}
impl Opcode for RVLookups {}

impl JoltProvableTrace for RVTraceRow {
  type JoltInstructionEnum = RVLookups;

  fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum> {
    // Handle fan-out 1-to-many
    // TODO(sragss): could handle this by instruction format instead.
    // TODO(sragss): Do we need to check that the result of the lookup is actually rd? Is this handeled by R1CS?
    match self.opcode {
      RV32IM::ADD => vec![RVLookups::ADD(ADDInstruction(
        self.rs1_val.unwrap(),
        self.rs2_val.unwrap(),
      ))],
      RV32IM::SUB => vec![RVLookups::SUB(SUBInstruction(
        self.rs1_val.unwrap(),
        self.rs2_val.unwrap(),
      ))],
      RV32IM::XOR => vec![RVLookups::XOR(XORInstruction(
        self.rs1_val.unwrap(),
        self.rs2_val.unwrap(),
      ))],
      RV32IM::OR => vec![RVLookups::OR(ORInstruction(
        self.rs1_val.unwrap(),
        self.rs2_val.unwrap(),
      ))],
      RV32IM::AND => vec![RVLookups::AND(ANDInstruction(
        self.rs1_val.unwrap(),
        self.rs2_val.unwrap(),
      ))],
      RV32IM::SLL => vec![RVLookups::SLL(SLLInstruction(
        self.rs1_val.unwrap(),
        self.rs2_val.unwrap(),
      ))],
      RV32IM::SRL => unimplemented!(),
      RV32IM::SRA => unimplemented!(),
      RV32IM::SLT => unimplemented!(),
      RV32IM::SLTU => unimplemented!(),
      RV32IM::ADDI => vec![RVLookups::ADD(ADDInstruction(
        self.rs1_val.unwrap(),
        self.imm.unwrap(),
      ))],
      RV32IM::XORI => vec![RVLookups::XOR(XORInstruction(
        self.rs1_val.unwrap(),
        self.imm.unwrap(),
      ))],
      RV32IM::ORI => vec![RVLookups::OR(ORInstruction(
        self.rs1_val.unwrap(),
        self.imm.unwrap(),
      ))],
      RV32IM::ANDI => vec![RVLookups::AND(ANDInstruction(
        self.rs1_val.unwrap(),
        self.imm.unwrap(),
      ))],
      RV32IM::SLLI => vec![RVLookups::SLL(SLLInstruction(
        self.rs1_val.unwrap(),
        self.imm.unwrap(),
      ))],
      RV32IM::SRLI => unimplemented!(),
      RV32IM::SRAI => unimplemented!(),
      RV32IM::SLTI => unimplemented!(),
      RV32IM::SLTIU => unimplemented!(),
      _ => unimplemented!(),
    }
  }

  fn to_ram_ops(&self) -> Vec<MemoryOp> {
    let instruction_type = self.opcode.instruction_type();

    let rs1_read = || MemoryOp::new_read(self.rs1.unwrap(), self.rs1_val.unwrap());
    let rs2_read = || MemoryOp::new_read(self.rs2.unwrap(), self.rs2_val.unwrap());
    let rd_write = || MemoryOp::Write(self.rd.unwrap(), self.rd_pre_val.unwrap(), self.rd_post_val.unwrap());

    let memory_bytes_before = |index: usize| self.memory_bytes_before.as_ref().unwrap()[index] as u64;
    let memory_bytes_after= |index: usize| self.memory_bytes_after.as_ref().unwrap()[index] as u64;

    match instruction_type {
      RV32InstructionFormat::R => vec![
        rs1_read(),
        rs2_read(),
        rd_write()
      ],
      RV32InstructionFormat::U => vec![
        rd_write(),
      ],
      RV32InstructionFormat::I => match self.opcode {
        RV32IM::ADDI | RV32IM::SLLI | RV32IM::SRLI | RV32IM::SRAI | RV32IM::ANDI | RV32IM::ORI | RV32IM::XORI | RV32IM::SLTI | RV32IM::SLTIU => vec![rs1_read(), rd_write()],
        RV32IM::LB | RV32IM::LBU => vec![
          rs1_read(), 
          rd_write(), 
          MemoryOp::Read(self.rs1_val.unwrap() + self.imm.unwrap(), memory_bytes_before(0))
        ],
        RV32IM::LH | RV32IM::LHU => vec![
          rs1_read(), 
          rd_write(),
          MemoryOp::Read(self.rs1_val.unwrap() + self.imm.unwrap(), memory_bytes_before(0)),
          MemoryOp::Read(self.rs1_val.unwrap() + self.imm.unwrap() + 1, memory_bytes_before(1)),
        ],
        RV32IM::LW => vec![
          rs1_read(), 
          rd_write(),
          MemoryOp::Read(self.rs1_val.unwrap() + self.imm.unwrap(), memory_bytes_before(0)),
          MemoryOp::Read(self.rs1_val.unwrap() + self.imm.unwrap() + 1, memory_bytes_before(1)),
          MemoryOp::Read(self.rs1_val.unwrap() + self.imm.unwrap() + 2, memory_bytes_before(2)),
          MemoryOp::Read(self.rs1_val.unwrap() + self.imm.unwrap() + 3, memory_bytes_before(3)),
        ],
        _ => panic!("shouldn't happen")
      },
      RV32InstructionFormat::S => match self.opcode {
        RV32IM::SB => vec![
          rs1_read(), 
          rs2_read(), 
          MemoryOp::Write(self.rs1_val.unwrap() + self.imm.unwrap(), memory_bytes_before(0), memory_bytes_after(0))
        ],
        RV32IM::SH => vec![
          rs1_read(),
          rs2_read(),
          MemoryOp::Write(self.rs1_val.unwrap() + self.imm.unwrap(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::Write(self.rs1_val.unwrap() + self.imm.unwrap() + 1, memory_bytes_before(1), memory_bytes_after(1)),
        ],
        RV32IM::SW => vec![
          rs1_read(),
          rs2_read(),
          MemoryOp::Write(self.rs1_val.unwrap() + self.imm.unwrap(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::Write(self.rs1_val.unwrap() + self.imm.unwrap() + 1, memory_bytes_before(1), memory_bytes_after(1)),
          MemoryOp::Write(self.rs1_val.unwrap() + self.imm.unwrap() + 2, memory_bytes_before(2), memory_bytes_after(2)),
          MemoryOp::Write(self.rs1_val.unwrap() + self.imm.unwrap() + 3, memory_bytes_before(3), memory_bytes_after(3)),
        ],
        _ => panic!("shouldn't happen")
      }
      _ => panic!("shouldn't happen"),
    }
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
