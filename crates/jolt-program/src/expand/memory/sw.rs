use super::*;

/// Lowers word store `SW` by updating the selected lane of an aligned doubleword.
///
/// The sequence proves word alignment, reads the containing doubleword, builds
/// a 32-bit lane mask, merges the low word of `rs2` into that lane, and writes
/// the whole doubleword back with `SD`.
pub(in crate::expand) fn expand_sw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    // Source `SW` requires word alignment even though the synthesized write is
    // a doubleword write to the containing aligned address.
    asm.emit_address(
        SourceInstructionKind::VirtualAssertWordAlignment,
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.emit_i(
        SourceInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.emit_i(
        SourceInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.emit_i(SourceInstructionKind::LD, v2.operand(), v1.operand(), 0);
    asm.emit_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    // v3 becomes a 32-bit lane mask shifted into place; v0 then carries the
    // shifted source word and finally the masked XOR delta.
    asm.emit_i(
        SourceInstructionKind::ORI,
        v3.operand(),
        reg(0),
        format_i_imm(-1),
    );
    asm.emit_i(SourceInstructionKind::SRLI, v3.operand(), v3.operand(), 32);
    asm.emit_r(
        SourceInstructionKind::SLL,
        v3.operand(),
        v3.operand(),
        v0.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::SLL,
        v0.operand(),
        reg(rs2(instruction)?),
        v0.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::XOR,
        v0.operand(),
        v2.operand(),
        v0.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::AND,
        v0.operand(),
        v0.operand(),
        v3.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::XOR,
        v2.operand(),
        v2.operand(),
        v0.operand(),
    );
    asm.emit_s(SourceInstructionKind::SD, v1.operand(), v2.operand(), 0);
    asm.release_many([v0, v1, v2, v3]);

    asm.finalize()
}
