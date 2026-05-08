use super::*;

pub(in crate::expand) fn expand_amoswapw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    super::shared::amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;
    super::shared::amo_post64(
        &mut asm,
        rs1(instruction)?,
        rs2(instruction)?,
        v_dword,
        v_shift,
        v_mask,
        rd(instruction)?,
        v_rd,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_mask)?;
    allocator.release(v_dword)?;
    allocator.release(v_shift)?;
    allocator.release(v_rd)?;
    Ok(sequence)
}
