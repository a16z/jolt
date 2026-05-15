use super::*;

/// Lowers `EBREAK` to a self-jump that terminates the emulator trace.
///
/// Jolt has no debugger trap target for `EBREAK`. The tracer treats unchanged
/// PC as program termination, so `JAL +0` preserves that observable behavior
/// while still producing a target-legal final row.
pub(in crate::expand) fn expand_ebreak(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let discard = asm.allocate()?;

    asm.emit_j(JoltInstructionKind::JAL, discard.operand(), 0);
    asm.release(discard);

    asm.finalize()
}
