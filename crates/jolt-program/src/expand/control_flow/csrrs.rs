use super::*;

pub(in crate::expand) fn expand_csrrs(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = virtual_register_for_csr(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm = ExpansionBuilder::new(*instruction);

    if rs1(instruction)? == 0 && rd(instruction)? == 0 {
        asm.emit_i(Kind::ADDI, reg(0), reg(0), 0);
        return asm.finalize();
    } else if rs1(instruction)? == 0 {
        asm.emit_i(Kind::ADDI, reg(rd(instruction)?), reg(virtual_reg), 0);
        return asm.finalize();
    } else if rd(instruction)? == 0 {
        asm.emit_r(
            Kind::OR,
            reg(virtual_reg),
            reg(virtual_reg),
            reg(rs1(instruction)?),
        );
        return asm.finalize();
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = asm.allocate()?;
        asm.emit_i(Kind::ADDI, temp.operand(), reg(rs1(instruction)?), 0);
        asm.emit_i(Kind::ADDI, reg(rd(instruction)?), reg(virtual_reg), 0);
        asm.emit_r(Kind::OR, reg(virtual_reg), reg(virtual_reg), temp.operand());
        asm.release(temp);
        return asm.finalize();
    }

    asm.emit_i(Kind::ADDI, reg(rd(instruction)?), reg(virtual_reg), 0);
    asm.emit_r(
        Kind::OR,
        reg(virtual_reg),
        reg(virtual_reg),
        reg(rs1(instruction)?),
    );

    asm.finalize()
}
