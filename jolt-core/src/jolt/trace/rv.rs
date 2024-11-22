use crate::jolt::instruction::and::ANDInstruction;
use crate::jolt::instruction::beq::BEQInstruction;
use crate::jolt::instruction::bge::BGEInstruction;
use crate::jolt::instruction::bgeu::BGEUInstruction;
use crate::jolt::instruction::bne::BNEInstruction;
use crate::jolt::instruction::mul::MULInstruction;
use crate::jolt::instruction::mulhu::MULHUInstruction;
use crate::jolt::instruction::mulu::MULUInstruction;
use crate::jolt::instruction::or::ORInstruction;
use crate::jolt::instruction::sll::SLLInstruction;
use crate::jolt::instruction::slt::SLTInstruction;
use crate::jolt::instruction::sltu::SLTUInstruction;
use crate::jolt::instruction::sra::SRAInstruction;
use crate::jolt::instruction::srl::SRLInstruction;
use crate::jolt::instruction::sub::SUBInstruction;
use crate::jolt::instruction::virtual_advice::ADVICEInstruction;
use crate::jolt::instruction::virtual_assert_aligned_memory_access::AssertAlignedMemoryAccessInstruction;
use crate::jolt::instruction::virtual_assert_lte::ASSERTLTEInstruction;
use crate::jolt::instruction::virtual_assert_valid_div0::AssertValidDiv0Instruction;
use crate::jolt::instruction::virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction;
use crate::jolt::instruction::virtual_assert_valid_unsigned_remainder::AssertValidUnsignedRemainderInstruction;
use crate::jolt::instruction::virtual_move::MOVEInstruction;
use crate::jolt::instruction::xor::XORInstruction;
use crate::jolt::instruction::{add::ADDInstruction, virtual_movsign::MOVSIGNInstruction};
use crate::jolt::vm::rv32i_vm::RV32I;
use common::rv_trace::{ELFInstruction, RVTraceRow, RV32IM};

impl TryFrom<&ELFInstruction> for RV32I {
    type Error = &'static str;

    #[rustfmt::skip] // keep matches pretty
    fn try_from(instruction: &ELFInstruction) -> Result<Self, Self::Error> {
        match instruction.opcode {
            RV32IM::ADD  => Ok(ADDInstruction::default().into()),
            RV32IM::SUB  => Ok(SUBInstruction::default().into()),
            RV32IM::XOR  => Ok(XORInstruction::default().into()),
            RV32IM::OR   => Ok(ORInstruction::default().into()),
            RV32IM::AND  => Ok(ANDInstruction::default().into()),
            RV32IM::SLL  => Ok(SLLInstruction::default().into()),
            RV32IM::SRL  => Ok(SRLInstruction::default().into()),
            RV32IM::SRA  => Ok(SRAInstruction::default().into()),
            RV32IM::SLT  => Ok(SLTInstruction::default().into()),
            RV32IM::SLTU => Ok(SLTUInstruction::default().into()),

            RV32IM::ADDI  => Ok(ADDInstruction::default().into()),
            RV32IM::XORI  => Ok(XORInstruction::default().into()),
            RV32IM::ORI   => Ok(ORInstruction::default().into()),
            RV32IM::ANDI  => Ok(ANDInstruction::default().into()),
            RV32IM::SLLI  => Ok(SLLInstruction::default().into()),
            RV32IM::SRLI  => Ok(SRLInstruction::default().into()),
            RV32IM::SRAI  => Ok(SRAInstruction::default().into()),
            RV32IM::SLTI  => Ok(SLTInstruction::default().into()),
            RV32IM::SLTIU => Ok(SLTUInstruction::default().into()),

            RV32IM::BEQ  => Ok(BEQInstruction::default().into()),
            RV32IM::BNE  => Ok(BNEInstruction::default().into()),
            RV32IM::BLT  => Ok(SLTInstruction::default().into()),
            RV32IM::BLTU => Ok(SLTUInstruction::default().into()),
            RV32IM::BGE  => Ok(BGEInstruction::default().into()),
            RV32IM::BGEU => Ok(BGEUInstruction::default().into()),

            RV32IM::JAL   => Ok(ADDInstruction::default().into()),
            RV32IM::JALR  => Ok(ADDInstruction::default().into()),
            RV32IM::AUIPC => Ok(ADDInstruction::default().into()),

            RV32IM::MUL => Ok(MULInstruction::default().into()),
            RV32IM::MULU => Ok(MULUInstruction::default().into()),
            RV32IM::MULHU => Ok(MULHUInstruction::default().into()),

            RV32IM::VIRTUAL_ADVICE => Ok(ADVICEInstruction::default().into()),
            RV32IM::VIRTUAL_MOVE => Ok(MOVEInstruction::default().into()),
            RV32IM::VIRTUAL_MOVSIGN => Ok(MOVSIGNInstruction::default().into()),
            RV32IM::VIRTUAL_ASSERT_EQ => Ok(BEQInstruction::default().into()),
            RV32IM::VIRTUAL_ASSERT_LTE => Ok(ASSERTLTEInstruction::default().into()),
            RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER => Ok(AssertValidUnsignedRemainderInstruction::default().into()),
            RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER => Ok(AssertValidSignedRemainderInstruction::default().into()),
            RV32IM::VIRTUAL_ASSERT_VALID_DIV0 => Ok(AssertValidDiv0Instruction::default().into()),
            RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT => Ok(AssertAlignedMemoryAccessInstruction::<32, 2>::default().into()),

            RV32IM::LW => Ok(AssertAlignedMemoryAccessInstruction::<32, 4>::default().into()),
            RV32IM::SW => Ok(AssertAlignedMemoryAccessInstruction::<32, 4>::default().into()),

            _ => Err("No corresponding RV32I instruction")
        }
    }
}

impl TryFrom<&RVTraceRow> for RV32I {
    type Error = &'static str;

    #[rustfmt::skip] // keep matches pretty
    fn try_from(row: &RVTraceRow) -> Result<Self, Self::Error> {
        match row.instruction.opcode {
            RV32IM::ADD => Ok(ADDInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SUB => Ok(SUBInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::XOR => Ok(XORInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::OR  => Ok(ORInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::AND => Ok(ANDInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SLL => Ok(SLLInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SRL => Ok(SRLInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SRA => Ok(SRAInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SLT  => Ok(SLTInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::SLTU => Ok(SLTUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),

            RV32IM::ADDI  => Ok(ADDInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::XORI  => Ok(XORInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::ORI   => Ok(ORInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::ANDI  => Ok(ANDInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::SLLI  => Ok(SLLInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::SRLI  => Ok(SRLInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::SRAI  => Ok(SRAInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::SLTI  => Ok(SLTInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::SLTIU => Ok(SLTUInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),

            RV32IM::BEQ  => Ok(BEQInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BNE  => Ok(BNEInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BLT  => Ok(SLTInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BLTU => Ok(SLTUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BGE  => Ok(BGEInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::BGEU => Ok(BGEUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),

            RV32IM::JAL  => Ok(ADDInstruction(row.instruction.address, row.imm_u32() as u64).into()),
            RV32IM::JALR => Ok(ADDInstruction(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::AUIPC => Ok(ADDInstruction(row.instruction.address, row.imm_u32() as u64).into()),

            RV32IM::MUL => Ok(MULInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::MULU => Ok(MULUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::MULHU => Ok(MULHUInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),

            RV32IM::VIRTUAL_ADVICE => Ok(ADVICEInstruction(row.advice_value.unwrap()).into()),
            RV32IM::VIRTUAL_MOVE => Ok(MOVEInstruction(row.register_state.rs1_val.unwrap()).into()),
            RV32IM::VIRTUAL_MOVSIGN => Ok(MOVSIGNInstruction(row.register_state.rs1_val.unwrap()).into()),
            RV32IM::VIRTUAL_ASSERT_EQ => Ok(BEQInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::VIRTUAL_ASSERT_LTE => Ok(ASSERTLTEInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER => Ok(AssertValidUnsignedRemainderInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER => Ok(AssertValidSignedRemainderInstruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::VIRTUAL_ASSERT_VALID_DIV0 => Ok(AssertValidDiv0Instruction(row.register_state.rs1_val.unwrap(), row.register_state.rs2_val.unwrap()).into()),
            RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT => Ok(AssertAlignedMemoryAccessInstruction::<32, 2>(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),

            RV32IM::LW => Ok(AssertAlignedMemoryAccessInstruction::<32, 4>(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),
            RV32IM::SW => Ok(AssertAlignedMemoryAccessInstruction::<32, 4>(row.register_state.rs1_val.unwrap(), row.imm_u32() as u64).into()),

            _ => Err("No corresponding RV32I instruction")
        }
    }
}
