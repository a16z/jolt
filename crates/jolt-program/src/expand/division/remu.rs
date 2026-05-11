use super::*;

pub(in crate::expand) fn expand_remu(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;

    asm.dispatch_j(JoltInstructionKind::VirtualAdvice, v0.operand(), 0);
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.dispatch_r(
        JoltInstructionKind::MUL,
        v0.operand(),
        v0.operand(),
        reg(rs2(instruction)?),
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertLTE,
        v0.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.dispatch_r(
        JoltInstructionKind::SUB,
        v0.operand(),
        reg(rs1(instruction)?),
        v0.operand(),
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.dispatch_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release(v0);

    asm.finalize()
}
