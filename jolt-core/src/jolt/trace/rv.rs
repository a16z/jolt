use super::JoltProvableTrace;
use crate::jolt::instruction::add::ADDInstruction;
use crate::jolt::instruction::and::ANDInstruction;
use crate::jolt::instruction::beq::BEQInstruction;
use crate::jolt::instruction::bge::BGEInstruction;
use crate::jolt::instruction::bgeu::BGEUInstruction;
use crate::jolt::instruction::bne::BNEInstruction;
use crate::jolt::instruction::or::ORInstruction;
use crate::jolt::instruction::sll::SLLInstruction;
use crate::jolt::instruction::slt::SLTInstruction;
use crate::jolt::instruction::sltu::SLTUInstruction;
use crate::jolt::instruction::sra::SRAInstruction;
use crate::jolt::instruction::srl::SRLInstruction;
use crate::jolt::instruction::sub::SUBInstruction;
use crate::jolt::instruction::xor::XORInstruction;
use crate::jolt::vm::read_write_memory::MemoryOp;
use crate::jolt::vm::{bytecode::BytecodeRow, rv32i_vm::RV32I};
use common::{ELFInstruction, MemoryState, RV32InstructionFormat, RVTraceRow, RV32IM};
use eyre::ensure;

impl TryFrom<&ELFInstruction> for RV32I {
    type Error = &'static str;

    #[rustfmt::skip] // keep matches pretty
    fn try_from(instruction: &ELFInstruction) -> Result<Self, Self::Error> {
        let result: Option<Self> = match instruction.opcode {
            RV32IM::ADD  => Some(ADDInstruction::default().into()),
            RV32IM::SUB  => Some(SUBInstruction::default().into()),
            RV32IM::XOR  => Some(XORInstruction::default().into()),
            RV32IM::OR   => Some(ORInstruction::default().into()),
            RV32IM::AND  => Some(ANDInstruction::default().into()),
            RV32IM::SLL  => Some(SLLInstruction::default().into()),
            RV32IM::SRL  => Some(SRLInstruction::default().into()),
            RV32IM::SRA  => Some(SRAInstruction::default().into()),
            RV32IM::SLT  => Some(SLTInstruction::default().into()),
            RV32IM::SLTU => Some(SLTUInstruction::default().into()),

            RV32IM::ADDI  => Some(ADDInstruction::default().into()),
            RV32IM::XORI  => Some(XORInstruction::default().into()),
            RV32IM::ORI   => Some(ORInstruction::default().into()),
            RV32IM::ANDI  => Some(ANDInstruction::default().into()),
            RV32IM::SLLI  => Some(SLLInstruction::default().into()),
            RV32IM::SRLI  => Some(SRLInstruction::default().into()),
            RV32IM::SRAI  => Some(SRAInstruction::default().into()),
            RV32IM::SLTI  => Some(SLTInstruction::default().into()),
            RV32IM::SLTIU => Some(SLTUInstruction::default().into()),

            RV32IM::BEQ  => Some(BEQInstruction::default().into()),
            RV32IM::BNE  => Some(BNEInstruction::default().into()),
            RV32IM::BLT  => Some(SLTInstruction::default().into()),
            RV32IM::BLTU => Some(SLTUInstruction::default().into()),
            RV32IM::BGE  => Some(BGEInstruction::default().into()),
            RV32IM::BGEU => Some(BGEUInstruction::default().into()),

            RV32IM::JAL   => Some(ADDInstruction::default().into()),
            RV32IM::JALR  => Some(ADDInstruction::default().into()),
            RV32IM::AUIPC => Some(ADDInstruction::default().into()),

            _ => None
        };

        if let Some(jolt_instruction) = result {
            Ok(jolt_instruction)
        } else {
            Err("No corresponding RV32I instruction")
        }
    }
}

impl From<&RVTraceRow> for RV32I {
    #[rustfmt::skip] // keep matches pretty
    fn from(row: &RVTraceRow) -> Self {
        match row.instruction.opcode {
            RV32IM::ADD => ADDInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::SUB => SUBInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::XOR => XORInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::OR  => ORInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::AND => ANDInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::SLL => SLLInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::SRL => SRLInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::SRA => SRAInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::SLT  => SLTInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::SLTU => SLTUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),

            RV32IM::ADDI  => ADDInstruction::<32>(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::XORI  => XORInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::ORI   => ORInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::ANDI  => ANDInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::SLLI  => SLLInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::SRLI  => SRLInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::SRAI  => SRAInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::SLTI  => SLTInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::SLTIU => SLTUInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),

            RV32IM::BEQ  => BEQInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::BNE  => BNEInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::BLT  => SLTInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::BLTU => SLTUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::BGE  => BGEInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),
            RV32IM::BGEU => BGEUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into(),

            RV32IM::JAL  => ADDInstruction(row.instruction.address, row.imm_u64()).into(),
            RV32IM::JALR => ADDInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into(),
            RV32IM::AUIPC => ADDInstruction(row.instruction.address, row.imm_u64()).into(),

            _ => panic!("Unsupported instruction")
        }
    }
}

impl JoltProvableTrace for RVTraceRow {
    type JoltInstructionEnum = RV32I;

    #[rustfmt::skip] // keep matches pretty
    fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum> {
        // Handle fan-out 1-to-many
        match self.instruction.opcode {
            RV32IM::ADD => vec![ADDInstruction::<32>(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::SUB => vec![SUBInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::XOR => vec![XORInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::OR  => vec![ORInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::AND => vec![ANDInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::SLL => vec![SLLInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::SRL => vec![SRLInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::SRA => vec![SRAInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::SLT  => vec![SLTInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::SLTU => vec![SLTUInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],

            RV32IM::ADDI  => vec![ADDInstruction::<32>(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::XORI  => vec![XORInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::ORI   => vec![ORInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::ANDI  => vec![ANDInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::SLLI  => vec![SLLInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::SRLI  => vec![SRLInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::SRAI  => vec![SRAInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::SLTI  => vec![SLTInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::SLTIU => vec![SLTUInstruction(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],

            RV32IM::BEQ  => vec![BEQInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::BNE  => vec![BNEInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::BLT  => vec![SLTInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::BLTU => vec![SLTUInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::BGE  => vec![BGEInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],
            RV32IM::BGEU => vec![BGEUInstruction(self.register_state.rs1_val.unwrap(), self.register_state.rs2_val.unwrap()).into()],

            RV32IM::JAL  => vec![ADDInstruction::<32>(self.instruction.address, self.imm_u64()).into()],
            RV32IM::JALR => vec![ADDInstruction::<32>(self.register_state.rs1_val.unwrap(), self.imm_u64()).into()],
            RV32IM::AUIPC => vec![ADDInstruction::<32>(self.instruction.address, self.imm_u64()).into()],

            _ => vec![]
        }
    }

    fn to_ram_ops(&self) -> Vec<MemoryOp> {
        let instruction_type = self.instruction.opcode.instruction_type();

        let rs1_read = || {
            MemoryOp::Read(
                self.instruction.rs1.unwrap(),
                self.register_state.rs1_val.unwrap(),
            )
        };
        let rs2_read = || {
            MemoryOp::Read(
                self.instruction.rs2.unwrap(),
                self.register_state.rs2_val.unwrap(),
            )
        };
        let rd_write = || {
            MemoryOp::Write(
                self.instruction.rd.unwrap(),
                self.register_state.rd_post_val.unwrap(),
            )
        };

        let ram_byte_read = |index: usize| match self.memory_state {
            Some(MemoryState::Read { address, value }) => (value >> (index * 8)) as u8,
            Some(MemoryState::Write {
                address,
                pre_value,
                post_value,
            }) => (pre_value >> (index * 8)) as u8,
            None => panic!("Memory state not found"),
        };
        let ram_byte_written = |index: usize| match self.memory_state {
            Some(MemoryState::Read { address, value }) => panic!("Unexpected MemoryState::Read"),
            Some(MemoryState::Write {
                address,
                pre_value,
                post_value,
            }) => (post_value >> (index * 8)) as u8,
            None => panic!("Memory state not found"),
        };

        let rs1_offset = || -> u64 {
            let rs1_val = self.register_state.rs1_val.unwrap();
            let imm = self.instruction.imm.unwrap();
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
                MemoryOp::no_op(),
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
            RV32InstructionFormat::I => match self.instruction.opcode {
                RV32IM::ADDI
                | RV32IM::SLLI
                | RV32IM::SRLI
                | RV32IM::SRAI
                | RV32IM::ANDI
                | RV32IM::ORI
                | RV32IM::XORI
                | RV32IM::SLTI
                | RV32IM::SLTIU => vec![
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
                    MemoryOp::Read(rs1_offset(), ram_byte_read(0) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::LH | RV32IM::LHU => vec![
                    rs1_read(),
                    MemoryOp::no_op(),
                    rd_write(),
                    MemoryOp::Read(rs1_offset(), ram_byte_read(0) as u64),
                    MemoryOp::Read(rs1_offset() + 1, ram_byte_read(1) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::LW => vec![
                    rs1_read(),
                    MemoryOp::no_op(),
                    rd_write(),
                    MemoryOp::Read(rs1_offset(), ram_byte_read(0) as u64),
                    MemoryOp::Read(rs1_offset() + 1, ram_byte_read(1) as u64),
                    MemoryOp::Read(rs1_offset() + 2, ram_byte_read(2) as u64),
                    MemoryOp::Read(rs1_offset() + 3, ram_byte_read(3) as u64),
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
                _ => unreachable!("{self:?}"),
            },
            RV32InstructionFormat::S => match self.instruction.opcode {
                RV32IM::SB => vec![
                    rs1_read(),
                    rs2_read(),
                    MemoryOp::no_op(),
                    MemoryOp::Write(rs1_offset(), ram_byte_written(0) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::SH => vec![
                    rs1_read(),
                    rs2_read(),
                    MemoryOp::no_op(),
                    MemoryOp::Write(rs1_offset(), ram_byte_written(0) as u64),
                    MemoryOp::Write(rs1_offset() + 1, ram_byte_written(1) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::SW => vec![
                    rs1_read(),
                    rs2_read(),
                    MemoryOp::no_op(),
                    MemoryOp::Write(rs1_offset(), ram_byte_written(0) as u64),
                    MemoryOp::Write(rs1_offset() + 1, ram_byte_written(1) as u64),
                    MemoryOp::Write(rs1_offset() + 2, ram_byte_written(2) as u64),
                    MemoryOp::Write(rs1_offset() + 3, ram_byte_written(3) as u64),
                ],
                _ => unreachable!(),
            },
            RV32InstructionFormat::UJ => vec![
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                rd_write(),
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
