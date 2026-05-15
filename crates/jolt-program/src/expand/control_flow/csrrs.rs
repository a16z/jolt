use super::*;

/// Lowers `CSRRS` to operations on Jolt's CSR virtual registers.
///
/// Supported CSRs are represented by reserved virtual registers. The sequence
/// must preserve the Zicsr read-before-write rule, including the special cases:
/// `rs1 = x0` is read-only, `rd = x0` discards the old CSR value, and
/// `rd == rs1` needs a temporary so the source bits are not overwritten before
/// the CSR update.
pub(in crate::expand) fn expand_csrrs(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = virtual_register_for_csr(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm = ExpansionBuilder::new(*instruction);

    if rs1(instruction)? == 0 {
        // Read-only `csrr rd, csr`: copy the CSR virtual register to rd.
        asm.emit_i(
            JoltInstructionKind::ADDI,
            reg(rd(instruction)?),
            reg(virtual_reg),
            0,
        );
        return asm.finalize();
    } else if rd(instruction)? == 0 {
        // Set-only `csrs csr, rs1`: update the CSR virtual register and
        // deliberately discard the old value.
        asm.emit_r(
            JoltInstructionKind::OR,
            reg(virtual_reg),
            reg(virtual_reg),
            reg(rs1(instruction)?),
        );
        return asm.finalize();
    } else if rd(instruction)? == rs1(instruction)? {
        // Preserve rs1 before writing rd with the old CSR value.
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
        asm.emit_r(
            JoltInstructionKind::OR,
            reg(virtual_reg),
            reg(virtual_reg),
            temp.operand(),
        );
        asm.release(temp);
        return asm.finalize();
    }

    // General case: rd receives the old CSR value, then the CSR accumulates
    // the source bits.
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        reg(virtual_reg),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::OR,
        reg(virtual_reg),
        reg(virtual_reg),
        reg(rs1(instruction)?),
    );

    asm.finalize()
}
