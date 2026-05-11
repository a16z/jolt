use super::*;

pub(in crate::expand) fn expand_lw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

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
    asm.dispatch_i(JoltInstructionKind::LD, v1.operand(), v1.operand(), 0);
    asm.dispatch_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.dispatch_r(
        SourceInstructionKind::SRL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.dispatch_i(
        JoltInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        v1.operand(),
        0,
    );
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}
