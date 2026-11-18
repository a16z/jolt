use super::*;

/// Lowers `EBREAK` to one `Panic` row, Jolt's failure halt.
///
/// Jolt's hart behaves as if an external debugger had set `dcsr.ebreakm`:
/// `EBREAK` in M-mode enters Debug Mode, where the hart is halted (RISC-V
/// Debug Specification, `dcsr.ebreakm`). Entering Debug Mode is not a trap,
/// so no trap CSR changes. The row is a table-free jump (lookup output 0)
/// that writes its link to a temporary; every tracer ends the run there.
pub(in crate::expand) fn expand_ebreak(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let discard = asm.allocate()?;

    asm.emit_i(Kind::JoltPanic, discard.operand(), reg(0), 0);
    asm.release(discard);

    asm.finalize()
}
