use super::*;

pub(in crate::expand) fn expand_mret(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mepc_vr = mepc_register();
    let mut asm = ExpansionBuilder::new(*instruction);
    let jalr_rd = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::JALR,
        jalr_rd.operand(),
        reg(mepc_vr),
        0,
    );
    asm.release(jalr_rd);

    asm.finalize()
}
