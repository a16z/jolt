use super::*;

pub(in crate::expand) fn expand_amoswapw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut ops = super::shared::amo_pre64_ops(rs1(instruction)?, v_rd, v_dword, v_shift);
    ops.extend(super::shared::amo_post64_ops(
        rs1(instruction)?,
        rs2(instruction)?,
        v_dword,
        v_shift,
        v_mask,
        rd(instruction)?,
        v_rd,
    ));
    ops.extend([
        grammar::ExpansionOp::Release(v_mask),
        grammar::ExpansionOp::Release(v_dword),
        grammar::ExpansionOp::Release(v_shift),
        grammar::ExpansionOp::Release(v_rd),
    ]);
    core::ExpansionState::new(allocator).materialize_ops(instruction, ops)
}
