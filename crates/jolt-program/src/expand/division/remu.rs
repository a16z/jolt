use super::*;

pub(in crate::expand) fn expand_remu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;

    asm.emit_j(
        SourceInstructionKind::VirtualAdvice(jolt_riscv::instructions::VirtualAdvice(())),
        v0.operand(),
        0,
    );
    asm.emit_b(
        SourceInstructionKind::VirtualAssertMulUNoOverflow,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        SourceInstructionKind::MUL,
        v0.operand(),
        v0.operand(),
        reg(rs2(instruction)?),
    );
    asm.emit_b(
        SourceInstructionKind::VirtualAssertLTE,
        v0.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_r(
        SourceInstructionKind::SUB,
        v0.operand(),
        reg(rs1(instruction)?),
        v0.operand(),
    );
    asm.emit_b(
        SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release(v0);

    asm.finalize()
}
