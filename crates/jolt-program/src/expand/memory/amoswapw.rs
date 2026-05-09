use super::*;

pub(in crate::expand) fn expand_amoswapw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    super::shared::expand_amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;
    super::shared::expand_amo_post64(
        &mut asm,
        super::shared::AmoPost64 {
            rs1: rs1(instruction)?,
            v_rs2: rs2(instruction)?,
            v_dword,
            v_shift,
            v_mask,
            rd: rd(instruction)?,
            v_rd,
        },
    )?;
    asm.release_many([v_mask, v_dword, v_shift, v_rd])?;

    asm.finalize()
}
