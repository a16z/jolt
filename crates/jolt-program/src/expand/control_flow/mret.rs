use super::*;

/// Lowers `MRET` to a jump through the proof-facing `mepc` virtual register.
///
/// Jolt currently models M-mode-only execution with no interrupt hardware. The
/// ZeroOS trampoline restores `mstatus` explicitly before `MRET`, so this
/// source instruction only needs to transfer control to `mepc`.
pub(in crate::expand) fn expand_mret(
    instruction: &SourceInstructionRow,
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
