use super::*;

pub(in crate::expand) fn expand_sw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    asm.dispatch_address(
        JoltInstructionKind::VirtualAssertWordAlignment,
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.dispatch_i(
        JoltInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.dispatch_i(
        JoltInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.dispatch_i(JoltInstructionKind::LD, v2.operand(), v1.operand(), 0);
    asm.dispatch_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.dispatch_i(
        JoltInstructionKind::ORI,
        v3.operand(),
        reg(0),
        format_i_imm(-1),
    );
    asm.dispatch_i(SourceInstructionKind::SRLI, v3.operand(), v3.operand(), 32);
    asm.dispatch_r(
        SourceInstructionKind::SLL,
        v3.operand(),
        v3.operand(),
        v0.operand(),
    );
    asm.dispatch_r(
        SourceInstructionKind::SLL,
        v0.operand(),
        reg(rs2(instruction)?),
        v0.operand(),
    );
    asm.dispatch_r(
        JoltInstructionKind::XOR,
        v0.operand(),
        v2.operand(),
        v0.operand(),
    );
    asm.dispatch_r(
        JoltInstructionKind::AND,
        v0.operand(),
        v0.operand(),
        v3.operand(),
    );
    asm.dispatch_r(
        JoltInstructionKind::XOR,
        v2.operand(),
        v2.operand(),
        v0.operand(),
    );
    asm.dispatch_s(JoltInstructionKind::SD, v1.operand(), v2.operand(), 0);
    asm.release_many([v0, v1, v2, v3]);

    asm.finalize()
}
