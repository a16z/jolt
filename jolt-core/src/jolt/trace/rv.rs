use ark_curve25519::Fr;
use ark_ff::PrimeField;
use eyre::ensure;

use crate::jolt::instruction::and::ANDInstruction;
use crate::jolt::instruction::beq::BEQInstruction;
use crate::jolt::instruction::bge::BGEInstruction;
use crate::jolt::instruction::bgeu::BGEUInstruction;
use crate::jolt::instruction::blt::BLTInstruction;
use crate::jolt::instruction::bltu::BLTUInstruction;
use crate::jolt::instruction::bne::BNEInstruction;
use crate::jolt::instruction::jal::JALInstruction;
use crate::jolt::instruction::jalr::JALRInstruction;
use crate::jolt::instruction::or::ORInstruction;
use crate::jolt::instruction::sll::SLLInstruction;
use crate::jolt::instruction::slt::SLTInstruction;
use crate::jolt::instruction::sltu::SLTUInstruction;
use crate::jolt::instruction::sra::SRAInstruction;
use crate::jolt::instruction::srl::SRLInstruction;
use crate::jolt::instruction::xor::XORInstruction;
use crate::jolt::instruction::{add::ADDInstruction, sub::SUBInstruction};
use crate::jolt::instruction::JoltInstruction;
use crate::jolt::vm::{pc::ELFRow, rv32i_vm::RV32I};
use common::{constants::REGISTER_COUNT, RV32InstructionFormat, RV32IM};

use super::JoltProvableTrace;
use crate::jolt::vm::read_write_memory::MemoryOp;

// TODO(sragss): Move upstream.
const C: usize = 4;
const M: usize = 1 << 16;

#[derive(Debug, Clone, PartialEq)]
pub struct RVTraceRow {
    pc: u64,
    opcode: RV32IM,

    rd: Option<u64>,
    rs1: Option<u64>,
    rs2: Option<u64>,

    imm: Option<u32>,

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
        imm: Option<u32>,
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
        res
    }

    // TODO(sragss): Hack. Move to common format and rm this conversion.
    fn from_common(common: common::RVTraceRow) -> Self {
        let mut memory_bytes_before = None;
        let mut memory_bytes_after = None;
        let trunc = |value: u64, position: usize| (value >> (position * 8)) as u8;

        if let Some(memory_state) = common.memory_state {
            memory_bytes_before = match common.instruction.opcode {
                RV32IM::LB | RV32IM::LBU | RV32IM::SB => match memory_state {
                    common::MemoryState::Read { address, value } => Some(vec![value as u8]),
                    common::MemoryState::Write {
                        address,
                        pre_value,
                        post_value,
                    } => Some(vec![pre_value as u8]),
                },
                RV32IM::LH | RV32IM::LHU | RV32IM::SH => match memory_state {
                    common::MemoryState::Read { address, value } => {
                        Some(vec![value as u8, trunc(value, 1)])
                    }
                    common::MemoryState::Write {
                        address,
                        pre_value,
                        post_value,
                    } => Some(vec![pre_value as u8, trunc(pre_value, 1)]),
                },
                RV32IM::LW | RV32IM::SW => match memory_state {
                    common::MemoryState::Read { address, value } => Some(vec![
                        value as u8,
                        trunc(value, 1),
                        trunc(value, 2),
                        trunc(value, 3),
                    ]),
                    common::MemoryState::Write {
                        address,
                        pre_value,
                        post_value,
                    } => Some(vec![
                        pre_value as u8,
                        trunc(pre_value, 1),
                        trunc(pre_value, 2),
                        trunc(pre_value, 3),
                    ]),
                },
                _ => panic!("memory_bytes_before shouldn't exist"),
            };

            memory_bytes_after = match common.instruction.opcode {
                RV32IM::LB | RV32IM::LBU | RV32IM::LH | RV32IM::LHU | RV32IM::LW => None,

                RV32IM::SB => match memory_state {
                    common::MemoryState::Write {
                        address,
                        pre_value,
                        post_value,
                    } => Some(vec![post_value as u8]),
                    _ => panic!("shouldn't happen"),
                },
                RV32IM::SH => match memory_state {
                    common::MemoryState::Write {
                        address,
                        pre_value,
                        post_value,
                    } => Some(vec![post_value as u8, trunc(post_value, 1)]),
                    _ => panic!("shouldn't happen"),
                },
                RV32IM::SW => match memory_state {
                    common::MemoryState::Write {
                        address,
                        pre_value,
                        post_value,
                    } => Some(vec![
                        post_value as u8,
                        trunc(post_value, 1),
                        trunc(post_value, 2),
                        trunc(post_value, 3),
                    ]),
                    _ => panic!("shouldn't happen"),
                },
                _ => panic!("memory_bytes_after shouldn't exist"),
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
            memory_bytes_before,
            memory_bytes_after,
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

    fn UType(
        pc: u64,
        opcode: RV32IM,
        rd: u64,
        rd_pre_val: u64,
        rd_post_val: u64,
        imm: u32,
    ) -> Self {
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
        let register_max: u64 = REGISTER_COUNT - 1;
        let register_value_max: u64 = (1 << 32) - 1;
        let assert_rd = || -> Result<(), eyre::Report> {
            ensure!(self.rd.is_some(), "Line {}: rd is None", line!());
            ensure!(
                self.rd_pre_val.is_some(),
                "Line {}: rd_pre_val is None",
                line!()
            );
            ensure!(
                self.rd_post_val.is_some(),
                "Line {}: rd_post_val is None",
                line!()
            );
            ensure!(
                self.rd.unwrap() <= register_max,
                "Line {}: rd is larger than register max",
                line!()
            );
            ensure!(
                self.rd_pre_val.unwrap() <= register_value_max,
                "Line {}: rd_pre_val is larger than register value max",
                line!()
            );
            ensure!(
                self.rd_post_val.unwrap() <= register_value_max,
                "Line {}: rd_post_val {} is larger than register max {}",
                line!(),
                self.rd_post_val.unwrap(),
                register_value_max
            );
            Ok(())
        };

        let assert_rs1 = || -> Result<(), eyre::Report> {
            ensure!(self.rs1.is_some(), "Line {}: rs1 is None", line!());
            ensure!(self.rs1_val.is_some(), "Line {}: rs1_val is None", line!());
            ensure!(
                self.rs1.unwrap() <= register_max,
                "Line {}: rs1 is larger than register max",
                line!()
            );
            ensure!(
                self.rs1_val.unwrap() <= register_value_max,
                "Line {}: rs1_val is larger than register value max",
                line!()
            );
            Ok(())
        };

        let assert_rs2 = || -> Result<(), eyre::Report> {
            ensure!(self.rs2.is_some(), "Line {}: rs2 is None", line!());
            ensure!(self.rs2_val.is_some(), "Line {}: rs2_val is None", line!());
            ensure!(
                self.rs2.unwrap() <= register_max,
                "Line {}: rs2 is larger than register max",
                line!()
            );
            ensure!(
                self.rs2_val.unwrap() <= register_value_max,
                "Line {}: rs2_val is larger than register value max",
                line!()
            );
            Ok(())
        };

        let assert_no_memory = || -> Result<(), eyre::Report> {
            ensure!(
                self.memory_bytes_before.is_none(),
                "Line {}: memory_bytes_before is not None",
                line!()
            );
            ensure!(
                self.memory_bytes_after.is_none(),
                "Line {}: memory_bytes_after is not None",
                line!()
            );
            Ok(())
        };

        // If imm is signed we check that (imm as i32) is in the range [-2^{bits-1}, 2^{bits-1} - 1]
        let assert_imm_signed = |imm_bits: usize| -> Result<(), eyre::Report> {
            ensure!(self.imm.is_some(), "Line {}: imm is None", line!());
            let imm_max: i32 = ((1u32 << (imm_bits - 1)) - 1) as i32;
            let imm_min: i32 = -1i32 * (1i32 << imm_bits - 1);
            ensure!(
                self.imm.unwrap() as i32 <= imm_max,
                "Line {}: imm is larger than imm max",
                line!()
            );
            ensure!(
                self.imm.unwrap() as i32 >= imm_min,
                "Line {}: imm is larger than imm min",
                line!()
            );
            Ok(())
        };

        // TODO(JOLT-71): Assert register addresses are in our preconfigured region.

        match self.opcode.instruction_type() {
            RV32InstructionFormat::R => {
                assert_rd()?;
                assert_rs1()?;
                assert_rs2()?;
                ensure!(self.imm.is_none(), "Line {}: imm is not None", line!());

                assert_no_memory()?;

                // Assert instruction correctness
                let lookups = self.to_jolt_instructions();
                assert_eq!(lookups.len(), 1);
                let expected_result: Fr = lookups[0].lookup_entry(C, M);
                let bigint = expected_result.into_bigint();
                let expected_result: u64 = bigint.0[0];

                ensure!(
                    expected_result == self.rd_post_val.unwrap(),
                    "Line {}: lookup result ({:?}) does not match rd_post_val {:?}",
                    line!(),
                    expected_result,
                    self.rd_post_val.unwrap()
                );
            }
            RV32InstructionFormat::I => {
                assert_rd()?;
                assert_rs1()?;
                ensure!(self.rs2.is_none(), "Line {}: rs2 is not None", line!());
                ensure!(
                    self.rs2_val.is_none(),
                    "Line {}: rs2_val is not None",
                    line!()
                );
                assert_imm_signed(12)?;

                // Arithmetic
                if self.opcode != RV32IM::LB
                    && self.opcode != RV32IM::LBU
                    && self.opcode != RV32IM::LHU
                    && self.opcode != RV32IM::LW
                {
                    assert_no_memory()?;

                    // Assert instruction correctness if arithmetic
                    let lookups = self.to_jolt_instructions();
                    assert!(lookups.len() == 1);
                    if lookups.len() == 1 && self.rd.unwrap() != 0 {
                        assert_eq!(lookups.len(), 1, "{self:?}");
                        let expected_result: Fr = lookups[0].lookup_entry(C, M);
                        let bigint = expected_result.into_bigint();
                        let expected_result: u64 = bigint.0[0];

                        if self.opcode != RV32IM::JALR {
                            ensure!(
                                expected_result == self.rd_post_val.unwrap(),
                                "Line {}: lookup result ({:?}) does not match rd_post_val {:?}",
                                line!(),
                                expected_result,
                                self.rd_post_val.unwrap()
                            );
                        } else {
                            // JALR
                            // TODO(JOLT-70): The next PC in the trace should be expected_result.
                            ensure!(
                                self.pc + 4 == self.rd_post_val.unwrap(),
                                "Line {}: JALR did not store PC + 4 to rd",
                                line!()
                            );
                        }
                    }
                }
            }
            RV32InstructionFormat::S => {
                ensure!(self.rd.is_none(), "Line {}: rd is not None", line!());
                ensure!(
                    self.rd_pre_val.is_none(),
                    "Line {}: rd_pre_val is not None",
                    line!()
                );
                ensure!(
                    self.rd_post_val.is_none(),
                    "Line {}: rd_post_val is not None",
                    line!()
                );
                assert_rs1()?;
                assert_rs2()?;
                assert_imm_signed(12)?;

                // Memory handled below
            }
            RV32InstructionFormat::SB => {
                ensure!(self.rd.is_none(), "Line {}: rd is not None", line!());
                ensure!(
                    self.rd_pre_val.is_none(),
                    "Line {}: rd_pre_val is not None",
                    line!()
                );
                ensure!(
                    self.rd_post_val.is_none(),
                    "Line {}: rd_post_val is not None",
                    line!()
                );

                assert_rs1()?;
                assert_rs2()?;

                assert_imm_signed(12)?;
            }
            RV32InstructionFormat::U => {
                assert_rd()?;
                assert_imm_signed(20)?;

                ensure!(self.rs1.is_none(), "Line {}: rs1 is not None", line!());
                ensure!(
                    self.rs1_val.is_none(),
                    "Line {}: rs1_val is not None",
                    line!()
                );
                ensure!(self.rs2.is_none(), "Line {}: rs2 is not None", line!());
                ensure!(
                    self.rs2_val.is_none(),
                    "Line {}: rs2_val is not None",
                    line!()
                );

                assert_no_memory()?;

                // Assert correct values in rd
                if self.rd.unwrap() != 0 {
                    match self.opcode {
                        RV32IM::LUI => {
                            ensure!(self.imm.is_some(), "Line {}: imm is None", line!());
                            let expected_rd = self.imm_u64(); // Load upper 20 bits
                            ensure!(
                                self.rd_post_val.unwrap() == expected_rd,
                                "Line {}: rd_post_val ({:b}) does not match expected_rd ({:b})",
                                line!(),
                                self.rd_post_val.unwrap(),
                                expected_rd
                            );
                        }
                        RV32IM::AUIPC => {
                            ensure!(self.imm.is_some(), "Line {}: imm is None", line!());
                            let expected_offset = self.imm_u64();
                            let expected_rd = expected_offset + self.pc;
                            ensure!(
                                self.rd_post_val.unwrap() == expected_rd,
                                "Line {}: rd_post_val does not match expected_rd",
                                line!()
                            );
                        }
                        _ => unreachable!(),
                    };
                } else {
                    ensure!(
                        self.rd_pre_val.unwrap() == 0,
                        "Line {}: rd_pre_val should be 0 for 0 register.",
                        line!()
                    );
                    ensure!(
                        self.rd_post_val.unwrap() == 0,
                        "Line {}: rd_post_val should be 0 for 0 register.",
                        line!()
                    );
                }
            }
            RV32InstructionFormat::UJ => {
                ensure!(self.opcode == RV32IM::JAL, "UJ was not JAL");
                ensure!(self.rs1.is_none(), "Line {}: rs1 is not None", line!());
                ensure!(
                    self.rs1_val.is_none(),
                    "Line {}: rs1_val is not None",
                    line!()
                );
                ensure!(self.rs2.is_none(), "Line {}: rs2 is not None", line!());
                ensure!(
                    self.rs2_val.is_none(),
                    "Line {}: rs2_val is not None",
                    line!()
                );

                ensure!(self.imm.is_some(), "Line {}: imm is None", line!());

                assert_rd()?;
                assert!(self.imm.is_some());

                if self.rd.unwrap() != 0 {
                    // TODO(JOLT-70): The next PC in the trace should be target_address.

                    // For UJ Instructions imm is an address offset, thus power of 2.
                    ensure!(
                        self.pc + 4 == self.rd_post_val.unwrap(),
                        "Line {}: JALR did not store PC + 4 to rd",
                        line!()
                    );
                } else {
                    ensure!(
                        self.rd_pre_val.unwrap() == 0,
                        "Line {}: rd_pre_val should be 0 for 0 register.",
                        line!()
                    );
                    ensure!(
                        self.rd_post_val.unwrap() == 0,
                        "Line {}: rd_post_val should be 0 for 0 register.",
                        line!()
                    );
                }
            }
        }

        // Check memory_before / memory_after
        let assert_load = |size: usize| -> Result<(), eyre::Report> {
            ensure!(
                self.memory_bytes_before.is_some(),
                "Line {}: memory_bytes_before is None",
                line!()
            );
            ensure!(
                self.memory_bytes_after.is_none(),
                "Line {}: memory_bytes_after is not None",
                line!()
            );
            ensure!(
                self.memory_bytes_before.as_ref().unwrap().len() == size,
                "Line {}: memory_bytes_before length does not match size",
                line!()
            );

            let mut rd = self.rd_post_val.unwrap();
            for i in 0..size {
                let expected_byte = rd as u8;
                ensure!(
                    expected_byte == self.memory_bytes_before.as_ref().unwrap()[i],
                    "Line {}: Byte {} does not match between rd ({}) and memory_bytes_before ({})",
                    line!(),
                    i,
                    expected_byte,
                    self.memory_bytes_before.as_ref().unwrap()[i]
                );

                rd = rd >> 8;
            }

            Ok(())
        };
        let assert_store = |size: usize| -> Result<(), eyre::Report> {
            ensure!(
                self.memory_bytes_before.is_some(),
                "Line {}: memory_bytes_before is None",
                line!()
            );
            ensure!(
                self.memory_bytes_after.is_some(),
                "Line {}: memory_bytes_after is None",
                line!()
            );
            ensure!(
                self.memory_bytes_before.as_ref().unwrap().len() == size,
                "Line {}: memory_bytes_before length does not match size",
                line!()
            );
            ensure!(
                self.memory_bytes_after.as_ref().unwrap().len() == size,
                "Line {}: memory_bytes_after length does not match size",
                line!()
            );

            // Check that memory_bytes_after == rs2
            let mut store_val = self.rs2_val.unwrap();
            for i in 0..size {
                let expected_byte = store_val as u8;
                ensure!(
                    expected_byte == self.memory_bytes_after.as_ref().unwrap()[i],
                    "Line {}: Byte {} does not match between rs2 ({}) and memory_bytes_after ({})",
                    line!(),
                    i,
                    expected_byte,
                    self.memory_bytes_after.as_ref().unwrap()[i]
                );
                store_val = store_val >> 8;
            }

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

    fn imm_u64(&self) -> u64 {
        match self.opcode.instruction_type() {
            RV32InstructionFormat::R => unimplemented!("R type does not use imm u64"),

            RV32InstructionFormat::I => self.imm.unwrap() as u64,

            RV32InstructionFormat::U => ((self.imm.unwrap() as u32) << 12u32) as u64,

            RV32InstructionFormat::S => unimplemented!("S type does not use imm u64"),

            // UJ-type instructions point to address offsets: even numbers.
            RV32InstructionFormat::UJ => (self.imm.unwrap() as u64) << 1u64,
            _ => unimplemented!(),
        }
    }
}

impl JoltProvableTrace for RVTraceRow {
    type JoltInstructionEnum = RV32I;

    #[rustfmt::skip] // keep matches pretty
    fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum> {
    // Handle fan-out 1-to-many

    // TODO(sragss): Do we need to check that the result of the lookup is actually rd? Is this handeled by R1CS?
    match self.opcode {
      RV32IM::ADD => vec![ADDInstruction::<32>(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SUB => vec![SUBInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::XOR => vec![XORInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::OR  => vec![ORInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::AND => vec![ANDInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SLL => vec![SLLInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SRL => vec![SRLInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SRA => vec![SRAInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SLT  => vec![SLTInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::SLTU => vec![SLTUInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],

      RV32IM::ADDI  => vec![ADDInstruction::<32>(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::XORI  => vec![XORInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::ORI   => vec![ORInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::ANDI  => vec![ANDInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::SLLI  => vec![SLLInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::SRLI  => vec![SRLInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::SRAI  => vec![SRAInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::SLTI  => vec![SLTInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],
      RV32IM::SLTIU => vec![SLTUInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],

      RV32IM::BEQ  => vec![BEQInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BNE  => vec![BNEInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BLT  => vec![BLTInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BLTU => vec![BLTUInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BGE  => vec![BGEInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],
      RV32IM::BGEU => vec![BGEUInstruction(self.rs1_val.unwrap(), self.rs2_val.unwrap()).into()],

      RV32IM::JAL  => vec![JALInstruction(self.pc, self.imm_u64()).into()],
      RV32IM::JALR => vec![JALRInstruction(self.rs1_val.unwrap(), self.imm_u64()).into()],

      RV32IM::AUIPC => vec![ADDInstruction::<32>(self.pc, self.imm_u64()).into()],

      _ => vec![]
    }
  }

    #[rustfmt::skip] // keep matches pretty
    fn to_ram_ops(&self) -> Vec<MemoryOp> {
    let instruction_type = self.opcode.instruction_type();

    let rs1_read = || MemoryOp::Read(self.rs1.unwrap(), self.rs1_val.unwrap());
    let rs2_read = || MemoryOp::Read(self.rs2.unwrap(), self.rs2_val.unwrap());
    let rd_write = || MemoryOp::Write(self.rd.unwrap(), self.rd_pre_val.unwrap(), self.rd_post_val.unwrap());

    let memory_bytes_before = |index: usize| self.memory_bytes_before.as_ref().unwrap()[index] as u64;
    let memory_bytes_after= |index: usize| self.memory_bytes_after.as_ref().unwrap()[index] as u64;

    let rs1_offset = || -> u64 {
      let rs1_val = self.rs1_val.unwrap();
      let imm = self.imm.unwrap();
      sum_u64_i32(rs1_val, imm as i32)
    };

    // Canonical ordering for memory instructions
    // 0: rs1
    // 1: rs2
    // 2: rd
    // 3: byte_0
    // 4: byte_1
    // 5: byte_2
    // 6: byte_3
    // If any are empty a no_op is inserted.

    // Validation: Number of ops should be a multiple of 7
    match instruction_type {
      RV32InstructionFormat::R => vec![
        rs1_read(),
        rs2_read(),
        rd_write(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op()
      ],
      RV32InstructionFormat::U => vec![
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        rd_write(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
      ],
      RV32InstructionFormat::I => match self.opcode {
        RV32IM::ADDI | RV32IM::SLLI | RV32IM::SRLI | RV32IM::SRAI | RV32IM::ANDI | RV32IM::ORI | RV32IM::XORI | RV32IM::SLTI | RV32IM::SLTIU => vec![
          rs1_read(),
          MemoryOp::no_op(),
          rd_write(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::LB | RV32IM::LBU => vec![
          rs1_read(),
          MemoryOp::no_op(),
          rd_write(),
          MemoryOp::Read(rs1_offset(), memory_bytes_before(0)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::LH | RV32IM::LHU => vec![
          rs1_read(),
          MemoryOp::no_op(),
          rd_write(),
          MemoryOp::Read(rs1_offset(), memory_bytes_before(0)),
          MemoryOp::Read(rs1_offset() + 1, memory_bytes_before(1)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::LW => vec![
          rs1_read(),
          MemoryOp::no_op(),
          rd_write(),
          MemoryOp::Read(rs1_offset(), memory_bytes_before(0)),
          MemoryOp::Read(rs1_offset() + 1, memory_bytes_before(1)),
          MemoryOp::Read(rs1_offset() + 2, memory_bytes_before(2)),
          MemoryOp::Read(rs1_offset() + 3, memory_bytes_before(3)),
        ],
        RV32IM::JALR => vec![
          rs1_read(),
          MemoryOp::no_op(),
          rd_write(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        _ => unreachable!("{self:?}")
      },
      RV32InstructionFormat::S => match self.opcode {
        RV32IM::SB => vec![
          rs1_read(),
          rs2_read(),
          MemoryOp::no_op(),
          MemoryOp::Write(rs1_offset(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::SH => vec![
          rs1_read(),
          rs2_read(),
          MemoryOp::no_op(),
          MemoryOp::Write(rs1_offset(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::Write(rs1_offset() + 1, memory_bytes_before(1), memory_bytes_after(1)),
          MemoryOp::no_op(),
          MemoryOp::no_op(),
        ],
        RV32IM::SW => vec![
          rs1_read(),
          rs2_read(),
          MemoryOp::no_op(),
          MemoryOp::Write(rs1_offset(), memory_bytes_before(0), memory_bytes_after(0)),
          MemoryOp::Write(rs1_offset() + 1, memory_bytes_before(1), memory_bytes_after(1)),
          MemoryOp::Write(rs1_offset() + 2, memory_bytes_before(2), memory_bytes_after(2)),
          MemoryOp::Write(rs1_offset() + 3, memory_bytes_before(3), memory_bytes_after(3)),
        ],
        _ => unreachable!()
      }
      RV32InstructionFormat::UJ => vec![
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        rd_write(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
      ],
      RV32InstructionFormat::SB => vec![
        rs1_read(),
        rs2_read(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
        MemoryOp::no_op(),
      ],
      _ => unreachable!("{self:?}"),
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

        let mut flags = vec![false; 15];

        flags[0] = match self.opcode {
            RV32IM::JAL | RV32IM::JALR | RV32IM::LUI | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[1] = match self.opcode {
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

        flags[2] = match self.opcode {
            RV32IM::LB | RV32IM::LH | RV32IM::LW | RV32IM::LBU | RV32IM::LHU => true,
            _ => false,
        };

        flags[3] = match self.opcode {
            RV32IM::SB | RV32IM::SH | RV32IM::SW => true,
            _ => false,
        };

        flags[4] = match self.opcode {
            RV32IM::JAL | RV32IM::JALR => true,
            _ => false,
        };

        flags[5] = match self.opcode {
            RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU => {
                true
            }
            _ => false,
        };

        // loads, stores, branches, jumps do not store the lookup output to rd (they may update rd in other ways)
        flags[6] = match self.opcode {
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

        flags[7] = match self.opcode {
            RV32IM::ADD | RV32IM::ADDI | RV32IM::JAL | RV32IM::JALR | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[8] = match self.opcode {
            RV32IM::SUB => true,
            _ => false,
        };

        flags[9] = match self.opcode {
            RV32IM::MUL | RV32IM::MULU | RV32IM::MULH | RV32IM::MULSU => true,
            _ => false,
        };

        // not incorporating advice instructions yet
        flags[10] = match self.opcode {
            _ => false,
        };

        // not incorporating assert true instructions yet
        flags[11] = match self.opcode {
            _ => false,
        };

        // not incorporating assert false instructions yet
        flags[12] = match self.opcode {
            _ => false,
        };

        let mask = 1u32 << 31;
        flags[13] = match self.imm {
            Some(imm) if imm & mask == mask => true,
            _ => false,
        };

        flags[14] = match self.opcode {
            RV32IM::LUI => true,
            _ => false,
        };

        flags
            .into_iter()
            .map(|bool_flag| bool_flag.into())
            .collect()
    }
}

fn sum_u64_i32(a: u64, b: i32) -> u64 {
    if b.is_negative() {
        let abs_b = b.abs() as u64;
        if a < abs_b {
            panic!("overflow")
        }
        a - abs_b
    } else {
        let b_u64: u64 = b.try_into().expect("failed u64 convesion");
        a + b_u64
    }
}

#[cfg(test)]
mod tests {
    use ark_curve25519::EdwardsProjective;
    use merlin::Transcript;

    use crate::lasso::memory_checking::MemoryCheckingVerifier;
    use crate::{
        jolt::vm::{
            instruction_lookups::InstructionLookupsProof, read_write_memory::ReadWriteMemory,
            rv32i_vm::RV32IJoltVM, Jolt,
        },
        poly::structured_poly::BatchablePolynomials,
        utils::{gen_random_point, math::Math, random::RandomTape},
    };

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
        let lui_row = RVTraceRow::UType(0, RV32IM::LUI, 10, 0, rd_update, imm);
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
            MemoryOp::Read(10, 15),
            MemoryOp::Read(11, 20),
            MemoryOp::Write(12, 0, 35),
            MemoryOp::no_op(),
            MemoryOp::no_op(),
            MemoryOp::no_op(),
            MemoryOp::no_op(),
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

        use common::path::JoltPaths;
        use common::serializable::Serializable;

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let mut num_errors = 0;
        for row in &converted_trace {
            if let Err(e) = row.validate() {
                // if row.opcode != RV32IM::SLLI {
                println!("Validation error: {} \n{:#?}\n\n", e, row);
                // }
                num_errors += 1;
            }
        }
        println!("Total errors: {num_errors}");
    }

    #[test]
    fn fib_e2e() {
        use crate::jolt::vm::rv32i_vm::RV32I;
        use crate::lasso::memory_checking::MemoryCheckingProver;
        use common::path::JoltPaths;
        use common::serializable::Serializable;

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let mut num_errors = 0;
        for row in &converted_trace {
            if let Err(e) = row.validate() {
                println!("Validation error: {} \n{:#?}\n\n", e, row);
                num_errors += 1;
            }
        }
        println!("Total errors: {num_errors}");

        // Prove lookups
        let lookup_ops: Vec<RV32I> = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| row.to_jolt_instructions())
            .collect();
        let r: Vec<Fr> = gen_random_point::<Fr>(lookup_ops.len().log_2());
        let mut prover_transcript = Transcript::new(b"example");
        let proof: InstructionLookupsProof<Fr, EdwardsProjective> =
            RV32IJoltVM::prove_instruction_lookups(lookup_ops, r.clone(), &mut prover_transcript);
        let mut verifier_transcript = Transcript::new(b"example");
        assert!(
            RV32IJoltVM::verify_instruction_lookups(proof, r, &mut verifier_transcript).is_ok()
        );

        // Prove memory
        const MEMORY_SIZE: usize = 1 << 16;

        // Emulator sets register 0xb to 0x1020 upon initialization for some reason,
        // something about Linux boot requiring it...
        let mut memory_ops: Vec<MemoryOp> = vec![MemoryOp::Write(11, 0, 4128)];
        memory_ops.extend(converted_trace.into_iter().flat_map(|row| row.to_ram_ops()));

        let next_power_of_two = memory_ops.len().next_power_of_two();
        memory_ops.resize(next_power_of_two, MemoryOp::no_op());

        let mut prover_transcript = Transcript::new(b"example");
        let (rw_memory, _): (ReadWriteMemory<Fr, EdwardsProjective>, _) =
            ReadWriteMemory::new(memory_ops, MEMORY_SIZE, &mut prover_transcript);
        let batched_polys = rw_memory.batch();
        let commitments = ReadWriteMemory::commit(&batched_polys);

        let mut random_tape = RandomTape::new(b"test_tape");
        let proof = rw_memory.prove_memory_checking(
            &rw_memory,
            &batched_polys,
            &commitments,
            &mut prover_transcript,
            &mut random_tape,
        );

        let mut verifier_transcript = Transcript::new(b"example");
        ReadWriteMemory::verify_memory_checking(proof, &commitments, &mut verifier_transcript)
            .expect("proof should verify");

        // Prove bytecode

        // Prove R1CS
    }
}

#[cfg(test)]
mod trace_validation_tests {
    use super::*;

    #[test]
    fn validate_r_type() {
        let add = RVTraceRow::RType(0, RV32IM::ADD, 1, 2, 3, 0, 13, 6, 7);
        assert!(add.validate().is_ok());

        let rd_missing = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADD,
            rd: None,
            rs1: Some(12),
            rs2: Some(13),
            imm: None,
            rd_pre_val: None,
            rd_post_val: None,
            rs1_val: Some(12),
            rs2_val: Some(12),
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(rd_missing.validate().is_err());

        let wrong_output = RVTraceRow {
            pc: 1,
            opcode: RV32IM::ADD,
            rd: Some(1),
            rs1: Some(2),
            rs2: Some(3),
            imm: None,
            rd_pre_val: Some(100_000),
            rd_post_val: Some(38),
            rs1_val: Some(30),
            rs2_val: Some(9),
            memory_bytes_before: None,
            memory_bytes_after: None,
        };

        assert!(wrong_output.validate().is_err());
    }

    #[test]
    fn validate_i_type_addi() {
        // Primary i_type test as it's the simplest
        // Ensures that normal instruction decoding works.
        // imm cannot be too big or too small (12-bit twos-complement).
        // rs2 / rs2_val should not be set.
        // instructions do indeed produce the correct output

        // Signed
        // ADDI postive
        let add_i = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(12),
            rd_pre_val: Some(12),
            rd_post_val: Some(25),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        let validation = add_i.validate();
        assert!(validation.is_ok());

        // ADDI negative (imm < rs1)
        let add_i_negative = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(-12i32 as u32),
            rd_pre_val: Some(12),
            rd_post_val: Some(1),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(add_i_negative.validate().is_ok());

        // ADDI negative (imm > rs1)
        let add_i_negative = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(-25i32 as u32),
            rd_pre_val: Some(12),
            rd_post_val: Some(-12i32 as u32 as u64),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(add_i_negative.validate().is_ok());

        // ADDI rs2 present
        let add_i_rs2_present = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: Some(10),
            imm: Some(-25i32 as u32),
            rd_pre_val: Some(12),
            rd_post_val: Some(-12i32 as u32 as u64),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(add_i_rs2_present.validate().is_err());

        // ADDI positive too big
        let imm_positive_max = (1u32 << 11) - 1;
        let add_i_imm_too_big = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(imm_positive_max + 1),
            rd_pre_val: Some(12),
            rd_post_val: Some((imm_positive_max + 1 + 13) as u64),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        let validation = add_i_imm_too_big.validate();
        assert!(validation.is_err(), "{:?}", validation.unwrap());
        let add_i_imm_good = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(imm_positive_max - 1),
            rd_pre_val: Some(12),
            rd_post_val: Some((imm_positive_max - 1 + 13) as u64),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        let validation = add_i_imm_good.validate();
        assert!(validation.is_ok());

        // ADDI negative too big
        let imm_max_negative = -1i32 * (1i32 << 11);
        let imm_too_negative: u32 = (imm_max_negative - 1) as u32;
        let add_i_imm_too_big = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(imm_too_negative),
            rd_pre_val: Some(12),
            rd_post_val: Some((imm_too_negative + 13).into()),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        let validation = add_i_imm_too_big.validate();
        assert!(validation.is_err());
        let add_i_imm_good = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(imm_too_negative + 1),
            rd_pre_val: Some(12),
            rd_post_val: Some((imm_too_negative + 1 + 13).into()),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        let validation = add_i_imm_good.validate();
        assert!(validation.is_ok());

        // ADDI incorrect output
        let add_i = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(12),
            rd_pre_val: Some(12),
            rd_post_val: Some(100),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        let validation = add_i.validate();
        assert!(validation.is_err());

        // ADDI include memory
        let add_i = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(12),
            rd_pre_val: Some(12),
            rd_post_val: Some(25),
            rs1_val: Some(13),
            rs2_val: None,
            memory_bytes_before: Some(vec![]),
            memory_bytes_after: None,
        };
        let validation = add_i.validate();
        assert!(validation.is_err());
    }

    #[test]
    fn validate_i_type_xori() {
        // Primary unisgned I type test.
        // Validates that imm is not too big nor too small.
        let xor_i = RVTraceRow {
            pc: 10,
            opcode: RV32IM::XORI,
            rd: Some(2),
            rs1: Some(3),
            rs2: None,
            imm: Some(0b000111u32),
            rd_pre_val: Some(1000_000),
            rd_post_val: Some(0b101101u64),
            rs1_val: Some(0b101010u64),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(xor_i.validate().is_ok());

        // imm too big
        let imm_max = (1u32 << 11) - 1;
        let imm_val = imm_max + 1;
        let rs1_val = 202;
        let rd_post_val = (imm_val as u64) ^ rs1_val;
        let xor_i = RVTraceRow {
            pc: 10,
            opcode: RV32IM::XORI,
            rd: Some(2),
            rs1: Some(3),
            rs2: None,
            imm: Some(imm_val),
            rd_pre_val: Some(1000_000),
            rd_post_val: Some(rd_post_val),
            rs1_val: Some(rs1_val),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(xor_i.validate().is_err());
        let imm_val = imm_max;
        let rs1_val = 202;
        let rd_post_val = (imm_val as u64) ^ rs1_val;
        let xor_i = RVTraceRow {
            pc: 10,
            opcode: RV32IM::XORI,
            rd: Some(2),
            rs1: Some(3),
            rs2: None,
            imm: Some(imm_val),
            rd_pre_val: Some(1000_000),
            rd_post_val: Some(rd_post_val),
            rs1_val: Some(rs1_val),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(xor_i.validate().is_ok());

        // imm wrong
        let xor_i = RVTraceRow {
            pc: 10,
            opcode: RV32IM::XORI,
            rd: Some(2),
            rs1: Some(3),
            rs2: None,
            imm: Some(0b000111u32),
            rd_pre_val: Some(1000_000),
            rd_post_val: Some(0b101100u64),
            rs1_val: Some(0b101010u64),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(xor_i.validate().is_err());
    }

    #[test]
    fn validate_i_type_loads() {
        let imm_max = (1u32 << 11) - 1;
        let imm_min: i32 = -1i32 * (1i32 << 11);
        let imm_min_u32 = imm_min as u32;
        // LB
        // imm positive
        let lb = RVTraceRow {
            pc: 20,
            opcode: RV32IM::LB,
            rd: Some(2),
            rs1: Some(2),
            rs2: None,
            imm: Some(imm_max),
            rd_pre_val: Some(12),
            rd_post_val: Some(12),
            rs1_val: Some(100_000),
            rs2_val: None,
            memory_bytes_before: Some(vec![12u8]),
            memory_bytes_after: None,
        };
        assert!(lb.validate().is_ok());

        // imm negative
        let lb = RVTraceRow {
            pc: 20,
            opcode: RV32IM::LB,
            rd: Some(2),
            rs1: Some(2),
            rs2: None,
            imm: Some(imm_min_u32),
            rd_pre_val: Some(12),
            rd_post_val: Some(100),
            rs1_val: Some(100_000),
            rs2_val: None,
            memory_bytes_before: Some(vec![100u8]),
            memory_bytes_after: None,
        };
        assert!(lb.validate().is_ok());

        // imm positive too big
        let lb = RVTraceRow {
            pc: 20,
            opcode: RV32IM::LB,
            rd: Some(2),
            rs1: Some(2),
            rs2: None,
            imm: Some(imm_max + 1),
            rd_pre_val: Some(12),
            rd_post_val: Some(12),
            rs1_val: Some(100_000),
            rs2_val: None,
            memory_bytes_before: Some(vec![100u8]),
            memory_bytes_after: None,
        };
        assert!(lb.validate().is_err());

        // imm negative too big
        let lb = RVTraceRow {
            pc: 20,
            opcode: RV32IM::LB,
            rd: Some(2),
            rs1: Some(2),
            rs2: None,
            imm: Some((imm_min - 1) as u32),
            rd_pre_val: Some(12),
            rd_post_val: Some(12),
            rs1_val: Some(100_000),
            rs2_val: None,
            memory_bytes_before: Some(vec![100u8]),
            memory_bytes_after: None,
        };
        assert!(lb.validate().is_err());
    }

    #[test]
    fn validate_jalr() {
        // imm positive
        let pc: u64 = 12;
        let next_pc: u64 = pc + 4;
        let rs1_val: u64 = 102;
        let imm_val: u32 = 202;
        let jalr = RVTraceRow {
            pc,
            opcode: RV32IM::JALR,
            rd: Some(1),
            rs1: Some(1),
            rs2: None,
            imm: Some(imm_val),
            rd_pre_val: Some(0),
            rd_post_val: Some(next_pc),
            rs1_val: Some(rs1_val + (imm_val as u64)),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(
            jalr.validate().is_ok(),
            "{:?}",
            jalr.validate().unwrap_err()
        );

        // imm negative
        let jalr = RVTraceRow {
            pc: 2147483728,
            opcode: RV32IM::JALR,
            rd: Some(1),
            rs1: Some(1),
            rs2: None,
            imm: Some(4294967248),
            rd_pre_val: Some(2147483724),
            rd_post_val: Some(2147483732),
            rs1_val: Some(2147483724),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(jalr.validate().is_ok());

        // wrong rd_post_val
        let pc: u64 = 12;
        let wrong_next_pc = pc + 8;
        let rs1_val: u64 = 102;
        let imm_val: u32 = 202;
        let jalr = RVTraceRow {
            pc,
            opcode: RV32IM::JALR,
            rd: Some(1),
            rs1: Some(1),
            rs2: None,
            imm: Some(imm_val),
            rd_pre_val: Some(0),
            rd_post_val: Some(wrong_next_pc),
            rs1_val: Some(rs1_val + (imm_val as u64)),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(jalr.validate().is_err());
    }

    #[test]
    fn valiate_jal() {
        let jal = RVTraceRow {
            pc: 2147483656,
            opcode: RV32IM::JAL,
            rd: Some(1),
            rs1: None,
            rs2: None,
            imm: Some(2048),
            rd_pre_val: Some(0),
            rd_post_val: Some(2147483660),
            rs1_val: None,
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(jal.validate().is_ok());
    }

    #[test]
    fn validate_slli() {
        let slli = RVTraceRow {
            pc: 2147485384,
            opcode: RV32IM::SLLI,
            rd: Some(16),
            rs1: Some(8),
            rs2: None,
            imm: Some(1),
            rd_pre_val: Some(3781453883),
            rd_post_val: Some(2892698072),
            rs1_val: Some(3593832684),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(
            slli.validate().is_ok(),
            "{:?}",
            slli.validate().unwrap_err()
        );
    }

    #[test]
    fn validate_xori() {
        let xori = RVTraceRow {
            pc: 2147487420,
            opcode: RV32IM::XORI,
            rd: Some(14),
            rs1: Some(5),
            rs2: None,
            imm: Some(4294967295),
            rd_pre_val: Some(2273806215),
            rd_post_val: Some(2105376125),
            rs1_val: Some(2189591170),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(
            xori.validate().is_ok(),
            "{:?}",
            xori.validate().unwrap_err()
        );
    }

    #[test]
    fn validate_lui() {
        let lui = RVTraceRow {
            pc: 2147484868,
            opcode: RV32IM::LUI,
            rd: Some(11),
            rs1: None,
            rs2: None,
            imm: Some(4294443009),
            rd_pre_val: Some(192),
            rd_post_val: Some(2147487744),
            rs1_val: None,
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(lui.validate().is_ok(), "{:?}", lui.validate().unwrap_err());
    }

    #[test]
    fn validate_u_type() {
        let mut lui = RVTraceRow {
            pc: 2147484868,
            opcode: RV32IM::LUI,
            rd: Some(11),
            rs1: None,
            rs2: None,
            imm: Some(4294443009),
            rd_pre_val: Some(192),
            rd_post_val: Some(2147487744),
            rs1_val: None,
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        // rs1 / rs1_val / rs2 / rs2_val should not be set
        assert!(lui.validate().is_ok());
        lui.rs1 = Some(12);
        assert!(lui.validate().is_err());
        lui.rs1 = None;
        lui.rs1_val = Some(12);
        assert!(lui.validate().is_err());
        lui.rs1_val = None;
        lui.rs2 = Some(12);
        assert!(lui.validate().is_err());
        lui.rs2 = None;
        lui.rs2_val = Some(12);
        assert!(lui.validate().is_err());
        lui.rs2_val = None;
        assert!(lui.validate().is_ok());

        let imm_max: u32 = (1u32 << 19) - 1;
        let imm_min: i32 = -1i32 * (1i32 << 19);

        // wrong rd_post_val
        let mut lui = RVTraceRow {
            pc: 12,
            opcode: RV32IM::LUI,
            rd: Some(11),
            rs1: None,
            rs2: None,
            imm: Some(imm_max),
            rd_pre_val: Some(10),
            rd_post_val: Some((imm_max as u64) << 12u64 + 1), // WRONG
            rs1_val: None,
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(lui.validate().is_err());

        // imm positive too big
        let mut lui = RVTraceRow {
            pc: 12,
            opcode: RV32IM::LUI,
            rd: Some(11),
            rs1: None,
            rs2: None,
            imm: Some(imm_max),
            rd_pre_val: Some(10),
            rd_post_val: Some((imm_max as u64) << 12u64),
            rs1_val: None,
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(lui.validate().is_ok());
        lui.imm = Some(imm_max + 1);
        lui.rd_post_val = Some(((imm_max + 1) as u64) << 12u64); // imm too big by 1
        assert!(lui.validate().is_err());

        // imm negative too big
        lui.imm = Some(imm_min as u32);
        lui.rd_post_val = Some(((imm_min as u32) << 12u32) as u64);
        assert!(lui.validate().is_ok(), "{:?}", lui.validate().unwrap_err());
        lui.imm = Some((imm_min - 1) as u32);
        lui.rd_post_val = Some((((imm_min - 1) as u32) << 12u32) as u64); // imm too negative by 1
        assert!(lui.validate().is_err());

        // AUIPC rd_post_val correctness

        // positive
        let pc: u64 = 100;
        let imm = imm_max;
        let rd_expected: u64 = (imm << 12u32) as u64 + pc;
        let mut auipc = RVTraceRow {
            pc,
            opcode: RV32IM::AUIPC,
            rd: Some(12),
            rs1: None,
            rs2: None,

            imm: Some(imm),

            rd_pre_val: Some(10),
            rd_post_val: Some(rd_expected),

            rs1_val: None,
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(auipc.validate().is_ok());
        auipc.rd_post_val = Some(auipc.rd_post_val.unwrap() + 1);
        assert!(auipc.validate().is_err());
    }

    #[test]
    fn validate_s_type() {
        // Check errors on correct number of memories
        let mut sb = RVTraceRow {
            pc: 44,
            opcode: RV32IM::SB,
            rd: None,
            rs1: Some(12),
            rs2: Some(13),
            imm: Some(10),
            rd_pre_val: None,
            rd_post_val: None,
            rs1_val: Some(80),
            rs2_val: Some(12),
            memory_bytes_before: Some(vec![0u8]),
            memory_bytes_after: Some(vec![12u8]),
        };
        assert!(sb.validate().is_ok());

        // SB wrong memory_bytes_before/after sizes
        sb.memory_bytes_before = None;
        assert!(sb.validate().is_err());
        sb.memory_bytes_before = Some(vec![]);
        assert!(sb.validate().is_err());
        sb.memory_bytes_before = Some(vec![0u8, 0u8]);
        assert!(sb.validate().is_err());
        sb.memory_bytes_before = Some(vec![12u8]);
        assert!(sb.validate().is_ok());
        sb.memory_bytes_after = None;
        assert!(sb.validate().is_err());
        sb.memory_bytes_after = Some(vec![]);
        assert!(sb.validate().is_err());
        sb.memory_bytes_after = Some(vec![0u8, 0u8]);
        assert!(sb.validate().is_err());
        sb.memory_bytes_after = Some(vec![12u8]);
        assert!(sb.validate().is_ok());

        // SB wrong memory_bytes_after
        sb.memory_bytes_after = Some(vec![10u8]);
        assert!(sb.validate().is_err());
        sb.memory_bytes_after = Some(vec![12u8]);
        assert!(sb.validate().is_ok());

        // SH wrong memory_bytes_before/after sizes
        let rs2_val = 0b00000011_00000010;
        let mut sh = RVTraceRow {
            pc: 44,
            opcode: RV32IM::SH,
            rd: None,
            rs1: Some(12),
            rs2: Some(13),
            imm: Some(10),
            rd_pre_val: None,
            rd_post_val: None,
            rs1_val: Some(80),
            rs2_val: Some(rs2_val),
            memory_bytes_before: Some(vec![0u8, 0u8]),
            memory_bytes_after: Some(vec![2u8, 3u8]),
        };
        assert!(sh.validate().is_ok());
        sh.memory_bytes_before = None;
        assert!(sh.validate().is_err());
        sh.memory_bytes_before = Some(vec![0u8]);
        assert!(sh.validate().is_err());
        sh.memory_bytes_before = Some(vec![0u8, 0u8, 0u8]);
        assert!(sh.validate().is_err());
        sh.memory_bytes_before = Some(vec![0u8, 0u8]);
        assert!(sh.validate().is_ok());

        sh.memory_bytes_after = None;
        assert!(sh.validate().is_err());
        sh.memory_bytes_after = Some(vec![2u8]);
        assert!(sh.validate().is_err());
        sh.memory_bytes_after = Some(vec![2u8, 3u8, 0u8]);
        assert!(sh.validate().is_err());
        sh.memory_bytes_after = Some(vec![2u8, 3u8]);
        assert!(sh.validate().is_ok());

        // SH wrong memory_bytes_after
        sh.memory_bytes_after = Some(vec![100, 100]);
        assert!(sh.validate().is_err());

        // SW wrong memory_bytes_before/after sizes
        let rs2_val = 0b00000111_00000100_00000011_00000010;
        let mut sw = RVTraceRow {
            pc: 44,
            opcode: RV32IM::SW,
            rd: None,
            rs1: Some(12),
            rs2: Some(13),
            imm: Some(10),
            rd_pre_val: None,
            rd_post_val: None,
            rs1_val: Some(80),
            rs2_val: Some(rs2_val),
            memory_bytes_before: Some(vec![0u8, 0u8, 0u8, 10u8]),
            memory_bytes_after: Some(vec![2u8, 3u8, 4u8, 7u8]),
        };
        assert!(sw.validate().is_ok());
        sw.memory_bytes_before = None;
        assert!(sw.validate().is_err());
        sw.memory_bytes_before = Some(vec![0u8, 0u8]);
        assert!(sw.validate().is_err());

        // SW wrong memory_bytes_after
        sw.memory_bytes_after = Some(vec![2u8, 3u8, 4u8, 10u8]); // WRONG
        assert!(sw.validate().is_err());
    }

    #[test]
    fn validate_register_too_big() {
        let mut addi = RVTraceRow {
            pc: 0,
            opcode: RV32IM::ADDI,
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: Some(25),
            rd_pre_val: Some(12),
            rd_post_val: Some(60),
            rs1_val: Some(35),
            rs2_val: None,
            memory_bytes_before: None,
            memory_bytes_after: None,
        };
        assert!(addi.validate().is_ok());

        let register_max: u64 = (1 << 5) - 1;
        addi.rd = Some(register_max);
        assert!(addi.validate().is_ok());
        addi.rd = Some(register_max + 1);
        assert!(addi.validate().is_err());
        addi.rd = Some(2);
        addi.rs1 = Some(register_max);
        assert!(addi.validate().is_ok());
        addi.rs1 = Some(register_max + 1);
        assert!(addi.validate().is_err());
    }
}
