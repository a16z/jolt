use super::*;

pub(in crate::expand) fn expand_ebreak(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let discard = asm.allocate()?;

    asm.emit_j(JoltInstructionKind::JAL, discard.operand(), 0);
    asm.release(discard);

    asm.finalize()
}
