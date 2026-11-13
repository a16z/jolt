use super::csr::emit_csr_swap;
use super::*;

/// Lowers `CSRRW` to operations on Jolt's CSR virtual registers.
///
/// The reserved virtual register for the CSR is the proof-facing source of
/// truth. `rd` receives the old CSR value unless `rd = x0` (in which case the
/// CSR is not read), and the CSR receives `rs1`, legalized as described in
/// [`super::csr`]. If `rd == rs1`, temporaries consume or preserve the source
/// before `rd` is overwritten with the old value.
pub(in crate::expand) fn expand_csrrw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = virtual_register_for_csr(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let rd = rd(instruction)?;
    let rs1 = rs1(instruction)?;
    let mut asm = ExpansionBuilder::new(*instruction);
    emit_csr_swap(&mut asm, csr, virtual_reg, rd, reg(rs1))?;
    asm.finalize()
}
