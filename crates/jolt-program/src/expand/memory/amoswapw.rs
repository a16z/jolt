use super::*;

pub(in crate::expand) fn expand_amoswapw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    super::shared::amo_pre64(
        &mut sequence,
        rs1(instruction)?,
        v_rd,
        v_dword,
        v_shift,
        allocator,
    )?;
    super::shared::amo_post64(
        &mut sequence,
        rs1(instruction)?,
        rs2(instruction)?,
        v_dword,
        v_shift,
        v_mask,
        rd(instruction)?,
        v_rd,
        allocator,
    )?;
    sequence.finish_releasing(allocator, [v_mask, v_dword, v_shift, v_rd])
}
