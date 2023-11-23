use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use eyre::{ensure, bail};
use strum_macros::{EnumCount, EnumIter, FromRepr};

use crate::jolt::instruction::add::ADD32Instruction;
use crate::jolt::instruction::and::ANDInstruction;
use crate::jolt::instruction::beq::BEQInstruction;
use crate::jolt::instruction::bne::BNEInstruction;
use crate::jolt::instruction::bge::BGEInstruction;
use crate::jolt::instruction::bgeu::BGEUInstruction;
use crate::jolt::instruction::bltu::BLTUInstruction;
use crate::jolt::instruction::jal::JALInstruction;
use crate::jolt::instruction::jalr::JALRInstruction;
use crate::jolt::instruction::sltu::SLTUInstruction;
use crate::jolt::instruction::or::ORInstruction;
use crate::jolt::instruction::sll::SLLInstruction;
use crate::jolt::instruction::slt::SLTInstruction;
use crate::jolt::instruction::sra::SRAInstruction;
use crate::jolt::instruction::srl::SRLInstruction;
use crate::jolt::instruction::blt::BLTInstruction;
use crate::jolt::instruction::xor::XORInstruction;
use crate::jolt::instruction::{add::ADDInstruction, sub::SUBInstruction};
use crate::jolt::instruction::{JoltInstruction, Opcode};
use crate::jolt::subtable::LassoSubtable;
use crate::jolt::vm::{pc::ELFRow, rv32i_vm::RV32I};
use common::{RV32IM, RV32InstructionFormat};

use super::{JoltProvableTrace, MemoryOp};


#[derive(Debug, Clone, PartialEq)]
struct RVTraceRow {
  pc: u64,
  opcode: RV32IM,

  rd: Option<u64>,
  rs1: Option<u64>,
  rs2: Option<u64>,

  imm: Option<i32>,

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
    imm: Option<i32>,
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
    res.validate().expect("validation failed");
    res
  }

  // TODO(sragss): Hack. Move to common format and rm this conversion.
  fn from_common(
    common: common::RVTraceRow
  ) -> Self {
    let mut memory_bytes_before = None;
    let mut memory_bytes_after = None;
    let trunc = |value: u64, position: usize| (value >> (position * 8)) as u8;

    if let Some(memory_state) = common.memory_state {
        memory_bytes_before = match common.instruction.opcode {
            RV32IM::LB | RV32IM::LBU | RV32IM::SB => match memory_state {
                common::MemoryState::Read { address, value } => Some(vec![value as u8]),
                common::MemoryState::Write { address, pre_value, post_value } => Some(vec![pre_value as u8])
            },
            RV32IM::LH | RV32IM::LHU | RV32IM::SH => match memory_state {
                common::MemoryState::Read { address, value } => Some(vec![value as u8, trunc(value, 1)]),
                common::MemoryState::Write { address, pre_value, post_value } => Some(vec![pre_value as u8, trunc(pre_value, 1)])
            },
            RV32IM::LW | RV32IM::SW => match memory_state {
                common::MemoryState::Read { address, value } => Some(vec![value as u8, trunc(value, 1), trunc(value, 2), trunc(value, 3)]),
                common::MemoryState::Write { address, pre_value, post_value } => Some(vec![pre_value as u8, trunc(pre_value, 1), trunc(pre_value, 2), trunc(pre_value, 3)])
            },
            _ => panic!("memory_bytes_before shouldn't exist")
          };

        memory_bytes_after = match common.instruction.opcode {
            RV32IM::LB | RV32IM::LBU | RV32IM::LH | RV32IM::LHU | RV32IM::LW => None,

            RV32IM::SB => match memory_state {
                common::MemoryState::Write { address, pre_value, post_value } => Some(vec![pre_value as u8]),
                _ => panic!("shouldn't happen")
            },
            RV32IM::SH => match memory_state {
                common::MemoryState::Write { address, pre_value, post_value } => Some(vec![pre_value as u8, trunc(pre_value, 1)]),
                _ => panic!("shouldn't happen")
            },
            RV32IM::SW => match memory_state {
                common::MemoryState::Write { address, pre_value, post_value } => Some(vec![pre_value as u8, trunc(pre_value, 1), trunc(pre_value, 2), trunc(pre_value, 3)]),
                _ => panic!("shouldn't happen")
            },
            _ => panic!("memory_bytes_after shouldn't exist")
          }
    }

    RVTraceRow {
        pc: common.instruction.address,
        opcode: common.instruction.opcode,
        rd: common.instruction.rd,
        rs1: common.instruction.rs1,
        rs2: common.instruction.rs2,
        imm: common.instruction.imm,
        rd_pre_val: common.register_state.rd_pre_val,
        rd_post_val: common.register_state.rd_post_val,
        rs1_val: common.register_state.rs1_val,
        rs2_val: common.register_state.rs2_val,
        memory_bytes_before: memory_bytes_before,
        memory_bytes_after: memory_bytes_after,
    }
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

  fn UType(pc: u64, opcode: RV32IM, rd: u64, rd_pre_val: u64, rd_post_val: u64, imm: i32) -> Self {
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

  #[must_use]
  fn validate(&self) -> Result<(), eyre::Report> {
    let register_bits = 5;
    let register_max: u64 = (1 << register_bits) - 1;
    let register_value_max: u64 = (1u64 << 32) - 1;
    let assert_rd = || -> Result<(), eyre::Report> {
      ensure!(self.rd.is_some(), "Line {}: rd is None", line!());
      ensure!(self.rd_pre_val.is_some(), "Line {}: rd_pre_val is None", line!());
      ensure!(self.rd_post_val.is_some(), "Line {}: rd_post_val is None", line!());
      ensure!(self.rd.unwrap() <= register_max, "Line {}: rd is larger than register max", line!());
      ensure!(self.rd_pre_val.unwrap() <= register_value_max, "Line {}: rd_pre_val is larger than register value max", line!());
      ensure!(self.rd_post_val.unwrap() <= register_value_max, "Line {}: rd_post_val {} is larger than register max {}", line!(), self.rd_post_val.unwrap(), register_value_max);
      Ok(())
    };

    let assert_rs1 = || -> Result<(), eyre::Report> {
      ensure!(self.rs1.is_some(), "Line {}: rs1 is None", line!());
      ensure!(self.rs1_val.is_some(), "Line {}: rs1_val is None", line!());
      ensure!(self.rs1.unwrap() <= register_max, "Line {}: rs1 is larger than register max", line!());
      ensure!(self.rs1_val.unwrap() <= register_value_max, "Line {}: rs1_val is larger than register value max", line!());
      Ok(())
    };

    let assert_rs2 = || -> Result<(), eyre::Report> {
      ensure!(self.rs2.is_some(), "Line {}: rs2 is None", line!());
      ensure!(self.rs2_val.is_some(), "Line {}: rs2_val is None", line!());
      ensure!(self.rs2.unwrap() <= register_max, "Line {}: rs2 is larger than register max", line!());
      ensure!(self.rs2_val.unwrap() <= register_value_max, "Line {}: rs2_val is larger than register value max", line!());
      Ok(())
    };

    let assert_no_memory = || -> Result<(), eyre::Report> {
      ensure!(self.memory_bytes_before.is_none(), "Line {}: memory_bytes_before is not None", line!());
      ensure!(self.memory_bytes_after.is_none(), "Line {}: memory_bytes_after is not None", line!());
      Ok(())
    };

    let assert_imm = |imm_bits: usize| -> Result<(), eyre::Report> {
      ensure!(self.imm.is_some(), "Line {}: imm is None", line!());
      let imm_max: i32 = (1 << imm_bits) - 1;
      ensure!(self.imm.unwrap() <= imm_max, "Line {}: imm is larger than imm max", line!());
      Ok(())
    };

    // TODO(sragss): Assert register addresses are in our preconfigured region.
    match self.opcode.instruction_type() {
      RV32InstructionFormat::R => {
        assert_rd()?;
        assert_rs1()?;
        assert_rs2()?;
        ensure!(self.imm.is_none(), "Line {}: imm is not None", line!());

        assert_no_memory()?;
      }
      RV32InstructionFormat::I => {
        assert_rd()?;
        assert_rs1()?;
        ensure!(self.rs2.is_none(), "Line {}: rs2 is not None", line!());
        ensure!(self.rs2_val.is_none(), "Line {}: rs2_val is not None", line!());
        assert_imm(12)?;

        assert_no_memory()?;
      },
      RV32InstructionFormat::S => {
        ensure!(self.rd.is_none(), "Line {}: rd is not None", line!());
        assert_rs1()?;
        assert_rs2()?;
        assert_imm(12)?;

        // Memory handled below
      },
      RV32InstructionFormat::SB => bail!("Not yet implemented"),
      RV32InstructionFormat::U => {
        assert_rd()?;
        assert_imm(20)?;

        ensure!(self.rs1.is_none(), "Line {}: rs1 is not None", line!());
        ensure!(self.rs1_val.is_none(), "Line {}: rs1_val is not None", line!());
        ensure!(self.rs2.is_none(), "Line {}: rs2 is not None", line!());
        ensure!(self.rs2_val.is_none(), "Line {}: rs2_val is not None", line!());

        assert_no_memory()?;

        // Assert correct values in rd
        match self.opcode {
          RV32IM::LUI => {
            ensure!(self.imm.is_some(), "Line {}: imm is None", line!());
            ensure!(self.imm.unwrap() >= 0, "Line {}: imm is negative", line!());
            let expected_rd = (self.imm.unwrap() as u64) << 12u64; // Load upper 20 bits
            ensure!(self.rd_post_val.unwrap() == expected_rd, "Line {}: rd_post_val does not match expected_rd", line!());
          },
          RV32IM::AUIPC => {
            ensure!(self.imm.is_some(), "Line {}: imm is None", line!());
            ensure!(self.imm.unwrap() >= 0, "Line {}: imm is negative", line!());
            let expected_offset = (self.imm.unwrap() as u64) << 12u64;
            let expected_rd = expected_offset + self.pc;
            ensure!(self.rd_post_val.unwrap() == expected_rd, "Line {}: rd_post_val does not match expected_rd", line!());
          },
          _ => unreachable!()
        };
      }
      RV32InstructionFormat::UJ => bail!("Not yet implemented"),
    }

    // Check memory_before / memory_after
    let assert_load = |size: usize| -> Result<(), eyre::Report> {
      ensure!(self.memory_bytes_before.is_some(), "Line {}: memory_bytes_before is None", line!());
      ensure!(self.memory_bytes_after.is_none(), "Line {}: memory_bytes_after is not None", line!());
      ensure!(self.memory_bytes_before.as_ref().unwrap().len() == size, "Line {}: memory_bytes_before length does not match size", line!());
      Ok(())
    };
    let assert_store= |size: usize| -> Result<(), eyre::Report> {
      ensure!(self.memory_bytes_before.is_some(), "Line {}: memory_bytes_before is None", line!());
      ensure!(self.memory_bytes_after.is_some(), "Line {}: memory_bytes_after is None", line!());
      ensure!(self.memory_bytes_before.as_ref().unwrap().len() == size, "Line {}: memory_bytes_before length does not match size", line!());
      ensure!(self.memory_bytes_after.as_ref().unwrap().len() == size, "Line {}: memory_bytes_after length does not match size", line!());
      Ok(())
    };
    match self.opcode {
      RV32IM::LB | RV32IM::LBU => assert_load(1)?,
      RV32IM::LH | RV32IM::LHU => assert_load(2)?,
      RV32IM::LW => assert_load(4)?,
      RV32IM::SB => assert_store(1)?,
      RV32IM::SH => assert_store(2)?,
      RV32IM::SW => assert_store(4)?,
      _ => {}
    };
    Ok(())
  }
}

#[repr(u8)]
#[derive(Copy, Clone, EnumIter, EnumCount)]
#[enum_dispatch(JoltInstruction)]
pub enum RV32Lookups {
  ADD(ADD32Instruction),
  SUB(SUBInstruction),
  XOR(XORInstruction),
  OR(ORInstruction),
  AND(ANDInstruction),
  SLL(SLLInstruction),
  SRL(SRLInstruction),
  SRA(SRAInstruction),
  // SLT(SLTInstruction),
  // SLTU(SLTUInstruction),
  BEQ(BEQInstruction),
  BNE(BNEInstruction),
  BLT(BLTInstruction),
  BLTU(BLTUInstruction),
  BGE(BGEInstruction),
  BGEU(BGEUInstruction),
  JAL(JALInstruction),
  JALR(JALRInstruction)
}
impl Opcode for RV32Lookups {}

impl JoltProvableTrace for RVTraceRow {
  type JoltInstructionEnum = RV32Lookups;

  #[rustfmt::skip] // keep matches pretty
  fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum> {
    // Handle fan-out 1-to-many

    let imm_u64 = || -> u64 {self.imm.unwrap().try_into().unwrap()};
    // TODO(sragss): Do we need to check that the result of the lookup is actually rd? Is this handeled by R1CS?
    match self.opcode {
      RV32IM::ADD => vec![ADDInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SUB => vec![SUBInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::XOR => vec![XORInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::OR  => vec![ORInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::AND => vec![ANDInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SLL => vec![SLLInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SRL => vec![SRLInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SRA => vec![SRAInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      // RV32IM::SLT  => vec![SLTInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      // RV32IM::SLTU => vec![SLTUInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],

      RV32IM::ADDI  => vec![ADDInstruction(self.rs1_val.unwrap(), imm_u64()).into()],
      RV32IM::XORI  => vec![XORInstruction(self.rs1_val.unwrap(), imm_u64()).into()],
      RV32IM::ORI   => vec![ORInstruction(self.rs1_val.unwrap(), imm_u64()).into()],
      RV32IM::ANDI  => vec![ANDInstruction(self.rs1_val.unwrap(), imm_u64()).into()],
      RV32IM::SLLI  => vec![SLLInstruction(self.rs1_val.unwrap(), imm_u64()).into()],
      RV32IM::SRLI  => vec![SRLInstruction(self.rs1_val.unwrap(), imm_u64()).into()],
      RV32IM::SRAI  => vec![SRAInstruction(self.rs1_val.unwrap(), imm_u64()).into()],
      // RV32IM::SLTI  => vec![SLTInstruction(self.rs1_val.unwrap(), self.imm.unwrap()).into()],
      // RV32IM::SLTIU => vec![SLTUInstruction(self.rs1_val.unwrap(), self.imm.unwrap()).into()],

      RV32IM::BEQ  => vec![BEQInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BNE  => vec![BNEInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BLT  => vec![BLTInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BLTU => vec![BLTUInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BGE  => vec![BGEInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BGEU => vec![BGEUInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::JAL  => vec![JALInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::JALR => vec![JALRInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],

      _ => unimplemented!(),
    }
  }

  #[rustfmt::skip] // keep matches pretty
  fn to_ram_ops(&self) -> Vec<MemoryOp> {
    let instruction_type = self.opcode.instruction_type();

    let rs1_read = || MemoryOp::new_read(self.rs1.unwrap(), self.rs1_val.unwrap());
    let rs2_read = || MemoryOp::new_read(self.rs2.unwrap(), self.rs2_val.unwrap());
    let rd_write = || MemoryOp::new_write(self.rd.unwrap(), self.rd_pre_val.unwrap(), self.rd_post_val.unwrap());

    let memory_bytes_before = |index: usize| self.memory_bytes_before.as_ref().unwrap()[index] as u64;
    let memory_bytes_after= |index: usize| self.memory_bytes_after.as_ref().unwrap()[index] as u64;

    let rs1_offset = || -> u64 {
      let rs1_val = self.rs1_val.unwrap();
      let imm = self.imm.unwrap();
      if imm.is_negative() {
          rs1_val - imm as u64
      } else {
          rs1_val + imm as u64
      }
    };

    // Canonical ordering for memory instructions
    // 0: rd
    // 1: rs1
    // 2: rs2
    // 3: byte_0
    // 4: byte_1
    // 5: byte_2
    // 6: byte_3
    // If any are empty a no_op is inserted.

    // TODO(sragss): Always pad to 7 / 11 with 0 address reads.
    // Validation: Number of ops should be a multiple of 7. Tests for consistency.
    match instruction_type {
      RV32InstructionFormat::R => vec![
        rd_write(),
        rs1_read(),
        rs2_read(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op()
      ],
      RV32InstructionFormat::U => vec![
        rd_write(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
      ],
      RV32InstructionFormat::I => match self.opcode {
        RV32IM::ADDI | RV32IM::SLLI | RV32IM::SRLI | RV32IM::SRAI | RV32IM::ANDI | RV32IM::ORI | RV32IM::XORI | RV32IM::SLTI | RV32IM::SLTIU => vec![
          rd_write(),
          rs1_read(), 
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::LB | RV32IM::LBU => vec![
          rd_write(), 
          rs1_read(), 
          MemoryOp::no_op(),
          MemoryOp::Read(rs1_offset(), memory_bytes_before(0)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::LH | RV32IM::LHU => vec![
          rd_write(),
          rs1_read(), 
          MemoryOp::no_op(),
          MemoryOp::Read(rs1_offset(), memory_bytes_before(0)),
          MemoryOp::Read(rs1_offset() + 1, memory_bytes_before(1)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::LW => vec![
          rd_write(),
          rs1_read(), 
          MemoryOp::no_op(),
          MemoryOp::Read(rs1_offset(), memory_bytes_before(0)),
          MemoryOp::Read(rs1_offset() + 1, memory_bytes_before(1)),
          MemoryOp::Read(rs1_offset() + 2, memory_bytes_before(2)),
          MemoryOp::Read(rs1_offset() + 3, memory_bytes_before(3)),
        ],
        _ => unreachable!()
      },
      RV32InstructionFormat::S => match self.opcode {
        RV32IM::SB => vec![
          MemoryOp::no_op(),
          rs1_read(), 
          rs2_read(), 
          MemoryOp::Write(rs1_offset(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::SH => vec![
          MemoryOp::no_op(),
          rs1_read(),
          rs2_read(),
          MemoryOp::Write(rs1_offset(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::Write(rs1_offset() + 1, memory_bytes_before(1), memory_bytes_after(1)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::SW => vec![
          MemoryOp::no_op(),
          rs1_read(),
          rs2_read(),
          MemoryOp::Write(rs1_offset(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::Write(rs1_offset() + 1, memory_bytes_before(1), memory_bytes_after(1)),
          MemoryOp::Write(rs1_offset() + 2, memory_bytes_before(2), memory_bytes_after(2)),
          MemoryOp::Write(rs1_offset() + 3, memory_bytes_before(3), memory_bytes_after(3)),
        ],
        _ => unreachable!()
      }
      _ => unreachable!(),
    }
  }

  fn to_pc_trace(&self) -> ELFRow {
    ELFRow::new(
      self.pc.try_into().unwrap(),
      self.opcode as u64,
      self.rd.unwrap_or(0),
      self.rs1.unwrap_or(0),
      self.rs2.unwrap_or(0),
      self.imm.unwrap_or(0) as u64, // imm is always cast to its 32-bit repr, signed or unsigned
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
    let flag_13 = match self.imm {
      Some(imm) if imm < 0 => true,
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
      F::from(flag_14),
    ]
  }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::convert;


  #[test]
  fn to_pc_trace() {
    // ADD
    let add_row = RVTraceRow::RType(2, RV32IM::ADD, 12, 10, 11, 0, 35, 15, 20);
    let add_pc = add_row.to_pc_trace();
    let expected_pc = ELFRow::new(2, RV32IM::ADD as u64, 12, 10, 11, 0u64);
    assert_eq!(add_pc, expected_pc);

    // LUI
    let imm = 20;
    let rd_update = 20 << 12;
    let lui_row  = RVTraceRow::UType(0, RV32IM::LUI, 10, 0, rd_update, imm);
    let lui_pc = lui_row.to_pc_trace();
    let expected_pc = ELFRow::new(0, RV32IM::LUI as u64, 10, 0, 0, 20u64);
    assert_eq!(lui_pc, expected_pc);
  }

  #[test]
  fn to_ram_ops() {
    // ADD
    let add_row = RVTraceRow::RType(2, RV32IM::ADD, 12, 10, 11, 0, 35, 15, 20);
    let add_ram_ops = add_row.to_ram_ops();
    assert_eq!(add_ram_ops.len(), 7);
    let expected_memory_ops = vec![
      MemoryOp::new_write(12, 0, 35),
      MemoryOp::new_read(10, 15),
      MemoryOp::new_read(11, 20),
      MemoryOp::no_op(),
      MemoryOp::no_op(),
      MemoryOp::no_op(),
      MemoryOp::no_op()
    ];
    assert_eq!(add_ram_ops, expected_memory_ops);
  }

  #[test]
  fn load_conversion() {
    // 1. Load common::RVTraceRow from file
    // 2. Convert via RVTraceRow::from_common
    // 3. Run validation
    // 4. ...
    // 5. Profit

    use common::serializable::Serializable;
    use common::path::JoltPaths;
    use std::env;
    use std::path::PathBuf;

    let trace_location = JoltPaths::trace_path("fibonacci");
    let loaded_trace: Vec<common::RVTraceRow> = Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location).expect("deserialization failed");

    let converted_trace: Vec<RVTraceRow> = loaded_trace.into_iter().map(|common| RVTraceRow::from_common(common)).collect();

    let mut num_errors = 0;
    for row in &converted_trace {
        if let Err(e) = row.validate() {
            println!("Validation error: {} \n{:#?}\n\n", e, row);
            num_errors += 1;
        }

        if num_errors > 10 {
            panic!("too many errors");
        }
    }
  }
}
