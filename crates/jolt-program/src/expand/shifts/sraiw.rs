use super::*;

/// Lowers `SRAIW` by first restoring the signed 32-bit source value.
///
/// Arithmetic word shifts operate on the sign-extended low word of `rs1`, not
/// on arbitrary high bits already present in the register. The final
/// `VirtualSignExtendWord` preserves the RV64 word-result contract.
pub(in crate::expand) fn expand_sraiw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs1 = asm.allocate()?;
    let shift = instruction.operands.imm & 0x1f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);

    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        ),
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
        JoltInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        ),
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );
    asm.release(v_rs1);

    asm.finalize()
}
