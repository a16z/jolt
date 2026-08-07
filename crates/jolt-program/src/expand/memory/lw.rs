use super::*;

/// Lowers signed word load `LW` through a word-lane mask.
pub(in crate::expand) fn expand_lw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_address(
        SourceInstructionKind::VirtualAssertWordAlignment,
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.expand_i(
        SourceInstructionKind::VirtualAlignAddr(jolt_riscv::instructions::VirtualAlignAddr(())),
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(SourceInstructionKind::LD, v0.operand(), v0.operand(), 0);
    asm.expand_i(
        SourceInstructionKind::VirtualLaneMaskW(jolt_riscv::instructions::VirtualLaneMaskW(())),
        v1.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_r(
        SourceInstructionKind::VirtualLaneExtractS(jolt_riscv::instructions::VirtualLaneExtractS(
            (),
        )),
        reg(rd(instruction)?),
        v0.operand(),
        v1.operand(),
    );
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}
