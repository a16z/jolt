use super::*;

pub(in crate::expand) fn expand_divu(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_j(SourceInstructionKind::VirtualAdvice, v0.operand(), 0);
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidDiv0,
        reg(rs2(instruction)?),
        v0.operand(),
        0,
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertMulUNoOverflow,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        v1.operand(),
        v0.operand(),
        reg(rs2(instruction)?),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        v1.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::SUB,
        v1.operand(),
        reg(rs1(instruction)?),
        v1.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
        v1.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release_many([v0, v1]);

    asm.finalize()
}
