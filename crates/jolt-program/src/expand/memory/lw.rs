use super::*;

pub(in crate::expand) fn expand_lw(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

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
    asm.expand_i(SourceInstructionKind::LD, v1.operand(), v1.operand(), 0);
    asm.expand_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_r(
        SourceInstructionKind::SRL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.expand_i(
        SourceInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        v1.operand(),
        0,
    );
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}
