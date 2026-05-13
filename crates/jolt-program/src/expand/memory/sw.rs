use super::*;

pub(in crate::expand) fn expand_sw(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    asm.expand_address(
        SourceInstructionKind::VirtualAssertWordAlignment,
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(
        SourceInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(SourceInstructionKind::LD, v2.operand(), v1.operand(), 0);
    asm.expand_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_i(
        SourceInstructionKind::ORI,
        v3.operand(),
        reg(0),
        format_i_imm(-1),
    );
    asm.expand_i(SourceInstructionKind::SRLI, v3.operand(), v3.operand(), 32);
    asm.expand_r(
        SourceInstructionKind::SLL,
        v3.operand(),
        v3.operand(),
        v0.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::SLL,
        v0.operand(),
        reg(rs2(instruction)?),
        v0.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::XOR,
        v0.operand(),
        v2.operand(),
        v0.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::AND,
        v0.operand(),
        v0.operand(),
        v3.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::XOR,
        v2.operand(),
        v2.operand(),
        v0.operand(),
    );
    asm.expand_s(SourceInstructionKind::SD, v1.operand(), v2.operand(), 0);
    asm.release_many([v0, v1, v2, v3]);

    asm.finalize()
}
