use super::*;

pub(in crate::expand) fn expand_sraiw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs1 = asm.allocate()?;
    let shift = instruction.operands.imm & 0x1f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);

    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        v_rs1.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSRAI,
        reg(rd(instruction)?),
        v_rs1.operand(),
        bitmask as i128,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );
    asm.release(v_rs1);

    asm.finalize()
}
