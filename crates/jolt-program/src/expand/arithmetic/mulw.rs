use super::*;

pub(in crate::expand) fn expand_mulw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.emit_r(
        JoltInstructionKind::MUL,
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
