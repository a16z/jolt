use super::*;

/// Lowers `EBREAK` to a self-jump that halts the hart.
///
/// Jolt's hart behaves as if an external debugger had set `dcsr.ebreakm`:
/// `EBREAK` in M-mode enters Debug Mode, where the hart is halted (RISC-V
/// Debug Specification, `dcsr.ebreakm`). Entering Debug Mode is not a trap,
/// so no trap CSR changes. The tracer treats an unchanged PC as termination,
/// so `JAL +0` ends the trace with a target-legal final row.
pub(in crate::expand) fn expand_ebreak(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let discard = asm.allocate()?;

    asm.emit_j(Kind::JAL, discard.operand(), 0);
    asm.release(discard);

    asm.finalize()
}
