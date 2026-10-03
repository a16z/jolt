use super::*;

pub(in crate::expand) fn expand_amoswapw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_mask = asm.allocate()?;
    let v_dword = asm.allocate()?;
    let v_shift = asm.allocate()?;
    let v_rd = asm.allocate()?;

    super::shared::expand_amo_pre64(
        &mut asm,
        reg(rs1(instruction)?),
        v_rd.operand(),
        v_dword.operand(),
        v_shift.operand(),
    )?;
    super::shared::expand_amo_post64(
        &mut asm,
        super::shared::AmoPost64 {
            rs1: reg(rs1(instruction)?),
            v_rs2: reg(rs2(instruction)?),
            v_dword: v_dword.operand(),
            v_shift: v_shift.operand(),
            v_mask: v_mask.operand(),
            rd: reg(rd(instruction)?),
            v_rd: v_rd.operand(),
        },
    )?;
    asm.release_many([v_mask, v_dword, v_shift, v_rd]);

    asm.finalize()
}
