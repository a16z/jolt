use super::*;

pub(in crate::expand) fn expand_sll(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_pow2 = asm.allocate()?;

    asm.dispatch_i(
        JoltInstructionKind::VirtualPow2,
        v_pow2.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::MUL,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        v_pow2.operand(),
    );
    asm.release(v_pow2);

    asm.finalize()
}
