use super::*;

pub(in crate::expand) fn expand_divu(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.dispatch_j(JoltInstructionKind::VirtualAdvice, v0.operand(), 0);
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertValidDiv0,
        reg(rs2(instruction)?),
        v0.operand(),
        0,
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.dispatch_r(
        JoltInstructionKind::MUL,
        v1.operand(),
        v0.operand(),
        reg(rs2(instruction)?),
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertLTE,
        v1.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.dispatch_r(
        JoltInstructionKind::SUB,
        v1.operand(),
        reg(rs1(instruction)?),
        v1.operand(),
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        v1.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.dispatch_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release_many([v0, v1]);

    asm.finalize()
}
