use super::*;

/// Lowers `SRLIW` by shifting the source word through the high half first.
///
/// Multiplying by `2^32` moves the low 32 bits into bits 63:32. A logical
/// right shift by `32 + shamt` then yields the zero-filled word result, after
/// which `VirtualSignExtendWord` applies RV64's required word sign extension.
pub(in crate::expand) fn expand_srliw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs1 = asm.allocate()?;
    let shift = (instruction.operands.imm & 0x1f) + 32;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        v_rs1.operand(),
        reg(rs1(instruction)?),
        1i128 << 32,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSRLI,
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
