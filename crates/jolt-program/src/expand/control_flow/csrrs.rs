use super::csr::emit_csr_set;
use super::*;

/// Lowers `CSRRS` to operations on Jolt's CSR virtual registers.
///
/// `rd` receives the old CSR value unless `rd = x0`, and the CSR receives
/// `csr | rs1`, legalized as described in [`super::csr`], unless `rs1 = x0`
/// (then the CSR is not written). See [`super::csr::emit_csr_set`]
/// for the special cases.
pub(in crate::expand) fn expand_csrrs(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = virtual_register_for_csr(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let rd = rd(instruction)?;
    let rs1 = rs1(instruction)?;
    let mut asm = ExpansionBuilder::new(*instruction);
    emit_csr_set(&mut asm, csr, virtual_reg, rd, reg(rs1))?;
    asm.finalize()
}
