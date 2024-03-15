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
use crate::jolt::vm::rv32i_vm::RV32I;
use common::rv_trace::{ELFInstruction, RVTraceRow, RV32IM};

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

impl TryFrom<&RVTraceRow> for RV32I {
    type Error = &'static str;

    #[rustfmt::skip] // keep matches pretty
    fn try_from(row: &RVTraceRow) -> Result<Self, Self::Error> {
        let result: Option<Self> = match row.instruction.opcode {
            RV32IM::ADD => Some(ADDInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SUB => Some(SUBInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::XOR => Some(XORInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::OR  => Some(ORInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::AND => Some(ANDInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SLL => Some(SLLInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SRL => Some(SRLInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SRA => Some(SRAInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SLT  => Some(SLTInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SLTU => Some(SLTUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),

            RV32IM::ADDI  => Some(ADDInstruction::<32>(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::XORI  => Some(XORInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::ORI   => Some(ORInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::ANDI  => Some(ANDInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::SLLI  => Some(SLLInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::SRLI  => Some(SRLInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::SRAI  => Some(SRAInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::SLTI  => Some(SLTInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::SLTIU => Some(SLTUInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),

            RV32IM::BEQ  => Some(BEQInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BNE  => Some(BNEInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BLT  => Some(SLTInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BLTU => Some(SLTUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BGE  => Some(BGEInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BGEU => Some(BGEUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),

            RV32IM::JAL  => Some(ADDInstruction(row.instruction.address, row.imm_u64()).into()),
            RV32IM::JALR => Some(ADDInstruction(row.register_state.rs1_val.unwrap(), row.imm_u64()).into()),
            RV32IM::AUIPC => Some(ADDInstruction(row.instruction.address, row.imm_u64()).into()),

            _ => None
        };

        if let Some(jolt_instruction) = result {
            Ok(jolt_instruction)
        } else {
            Err("No corresponding RV32I instruction")
        }
    }
}
