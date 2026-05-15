use super::*;

/// Lowers `CSRRW` to operations on Jolt's CSR virtual registers.
///
/// The reserved virtual register for the CSR is the proof-facing source of
/// truth. The sequence preserves the read-before-write swap rule: `rd`
/// receives the old CSR value unless `rd = x0`, and the CSR virtual register
/// receives `rs1`. If `rd == rs1`, a temporary keeps the new CSR value alive
/// while `rd` is overwritten with the old one.
pub(in crate::expand) fn expand_csrrw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = virtual_register_for_csr(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm = ExpansionBuilder::new(*instruction);

    if rd(instruction)? == 0 {
        // `csrw csr, rs1`: write the CSR and discard the old value.
        asm.emit_i(
            JoltInstructionKind::ADDI,
            reg(virtual_reg),
            reg(rs1(instruction)?),
            0,
        );
        return asm.finalize();
    } else if rd(instruction)? == rs1(instruction)? {
        // Preserve rs1 before rd is overwritten with the old CSR value.
        let temp = asm.allocate()?;
        asm.emit_i(
            JoltInstructionKind::ADDI,
            temp.operand(),
            reg(rs1(instruction)?),
            0,
        );
        asm.emit_i(
            JoltInstructionKind::ADDI,
            reg(rd(instruction)?),
            reg(virtual_reg),
            0,
        );
        asm.emit_i(
            JoltInstructionKind::ADDI,
            reg(virtual_reg),
            temp.operand(),
            0,
        );
        asm.release(temp);
        return asm.finalize();
    }

    // General case: copy old CSR to rd, then copy rs1 into the CSR.
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        reg(virtual_reg),
        0,
    );
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(virtual_reg),
        reg(rs1(instruction)?),
        0,
    );

    asm.finalize()
}
