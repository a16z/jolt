//! Zicsr immediates are zero-extended five-bit constants. Legalization can
//! therefore be folded into the emitted immediate; no operand row is needed.

use super::csr::{emit_csr_read, CsrBitUpdate, MSTATUS_WRITABLE};
use super::*;
use jolt_riscv::SourceInstructionKind as Kind;

/// Lowers `CSRRWI`, suppressing the read for `rd = x0`.
pub(in crate::expand) fn expand_csrrwi(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = virtual_register_for_csr(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let rd = rd(instruction)?;
    let uimm = i128::from(rs1(instruction)?);
    let mut asm = ExpansionBuilder::new(*instruction);
    if rd != 0 {
        emit_csr_read(&mut asm, csr, virtual_reg, reg(rd));
    }
    match csr {
        CSR_MSTATUS => asm.emit_i(
            Kind::ADDI,
            reg(virtual_reg),
            reg(0),
            uimm & MSTATUS_WRITABLE,
        ),
        CSR_MEPC => asm.emit_i(Kind::ADDI, reg(virtual_reg), reg(0), uimm & !1),
        CSR_MTVEC if uimm & 2 != 0 => {
            asm.emit_i(Kind::ANDI, reg(virtual_reg), reg(virtual_reg), 1);
            if uimm & !3 != 0 {
                asm.emit_i(Kind::ORI, reg(virtual_reg), reg(virtual_reg), uimm & !3);
            }
        }
        _ => asm.emit_i(Kind::ADDI, reg(virtual_reg), reg(0), uimm),
    }
    asm.finalize()
}

pub(in crate::expand) fn expand_csrrsi(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    expand_bit_update(instruction, CsrBitUpdate::Set)
}

pub(in crate::expand) fn expand_csrrci(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    expand_bit_update(instruction, CsrBitUpdate::Clear)
}

fn expand_bit_update(
    instruction: &SourceInstructionRow,
    update: CsrBitUpdate,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = virtual_register_for_csr(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let rd = rd(instruction)?;
    let uimm = i128::from(rs1(instruction)?);
    let mask = match csr {
        CSR_MSTATUS => uimm & MSTATUS_WRITABLE,
        CSR_MEPC => uimm & !1,
        CSR_MTVEC => match update {
            CsrBitUpdate::Set if uimm & 2 != 0 => uimm & !3,
            CsrBitUpdate::Set | CsrBitUpdate::Clear => uimm & !2,
        },
        _ => uimm,
    };
    let mut asm = ExpansionBuilder::new(*instruction);
    if rd != 0 {
        emit_csr_read(&mut asm, csr, virtual_reg, reg(rd));
    }
    if mask != 0 {
        let (kind, immediate) = match update {
            CsrBitUpdate::Set => (Kind::ORI, mask),
            CsrBitUpdate::Clear => (Kind::ANDI, format_i_imm(!mask)),
        };
        asm.emit_i(kind, reg(virtual_reg), reg(virtual_reg), immediate);
    } else if rd == 0 {
        asm.emit_i(Kind::ADDI, reg(0), reg(0), 0);
    }
    asm.finalize()
}
