use ark_ff::PrimeField;
use common::MemoryState;
use eyre::ensure;

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
use common::{RV32InstructionFormat, RVTraceRow, RV32IM};

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

    fn to_bytecode_trace(&self) -> BytecodeRow {
        BytecodeRow::new(
            self.instruction.address.try_into().unwrap(),
            self.instruction.opcode as u64,
            self.instruction.rd.unwrap_or(0),
            self.instruction.rs1.unwrap_or(0),
            self.instruction.rs2.unwrap_or(0),
            self.instruction.imm.unwrap_or(0) as u64, // imm is always cast to its 32-bit repr, signed or unsigned
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
        // 14: Is concat (Note: used to be is_lui)
        // Arasu: Extra to get things working
        // 15: is lui or auipc
        // 16: is jal

        let mut flags = vec![false; 17];

        flags[0] = match self.instruction.opcode {
            RV32IM::JAL | RV32IM::LUI | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[1] = match self.instruction.opcode {
            RV32IM::ADDI
            | RV32IM::XORI
            | RV32IM::ORI
            | RV32IM::ANDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::AUIPC
            | RV32IM::JAL
            | RV32IM::JALR => true,
            _ => false,
        };

        flags[2] = match self.instruction.opcode {
            RV32IM::LB | RV32IM::LH | RV32IM::LW | RV32IM::LBU | RV32IM::LHU => true,
            _ => false,
        };

        flags[3] = match self.instruction.opcode {
            RV32IM::SB | RV32IM::SH | RV32IM::SW => true,
            _ => false,
        };

        flags[4] = match self.instruction.opcode {
            RV32IM::JAL | RV32IM::JALR => true,
            _ => false,
        };

        flags[5] = match self.instruction.opcode {
            RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU => {
                true
            }
            _ => false,
        };

        // loads, stores, branches, jumps do not store the lookup output to rd (they may update rd in other ways)
        flags[6] = match self.instruction.opcode {
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

        flags[7] = match self.instruction.opcode {
            RV32IM::ADD | RV32IM::ADDI | RV32IM::JAL | RV32IM::JALR | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[8] = match self.instruction.opcode {
            RV32IM::SUB => true,
            _ => false,
        };

        flags[9] = match self.instruction.opcode {
            RV32IM::MUL | RV32IM::MULU | RV32IM::MULH | RV32IM::MULSU => true,
            _ => false,
        };

        // TODO(JOLT-29): Used in the 'M' extension
        flags[10] = match self.instruction.opcode {
            _ => false,
        };

        // TODO(JOLT-29): Used in the 'M' extension
        flags[11] = match self.instruction.opcode {
            _ => false,
        };

        // TODO(JOLT-29): Used in the 'M' extension
        flags[12] = match self.instruction.opcode {
            _ => false,
        };

        let mask = 1u32 << 31;
        flags[13] = match self.instruction.imm {
            Some(imm) if imm & mask == mask => true,
            _ => false,
        };

        flags[14] = match self.instruction.opcode {
            RV32IM::XOR
            | RV32IM::XORI
            | RV32IM::OR
            | RV32IM::ORI
            | RV32IM::AND
            | RV32IM::ANDI
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU => true,
            _ => false,
        };

        flags[15] = match self.instruction.opcode {
            RV32IM::LUI | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[16] = match self.instruction.opcode {
            RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI => true,
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
