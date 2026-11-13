use super::csr::MSTATUS_MPIE;
use super::*;
use jolt_riscv::SourceInstructionKind as Kind;

/// Lowers `MRET` into M-mode trap return.
///
/// On an M-only hart `MRET` sets `mstatus.MIE := mstatus.MPIE`,
/// `mstatus.MPIE := 1` and `pc := mepc`; MPP stays M and the privilege mode
/// stays M. The `mstatus` virtual register holds only MIE (bit 3) and MPIE
/// (bit 7), so `mstatus >> 4` is MPIE moved to bit 3 and the update is
/// `mstatus := (mstatus >> 4) | MPIE`. `mepc` always has bit 0 clear, so the
/// `JALR` target is `mepc` itself. The reservation is kept: the privileged
/// specification allows but does not require `xRET` to clear it.
pub(in crate::expand) fn expand_mret(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mepc_vr = mepc_register();
    let mstatus_vr = mstatus_register();
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(Kind::SRLI, reg(mstatus_vr), reg(mstatus_vr), 4);
    asm.emit_i(Kind::ORI, reg(mstatus_vr), reg(mstatus_vr), MSTATUS_MPIE);

    let jalr_rd = asm.allocate()?;
    asm.emit_i(Kind::JALR, jalr_rd.operand(), reg(mepc_vr), 0);
    asm.release(jalr_rd);

    asm.finalize()
}
