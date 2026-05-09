use super::*;

pub(in crate::expand) fn expand_addw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_r(
        JoltInstructionKind::ADD,
        rd(instruction)?,
        rs1(instruction)?,
        rs2(instruction)?,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );

    asm.finalize()
}
