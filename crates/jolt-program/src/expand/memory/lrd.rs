use super::*;

pub(in crate::expand) fn expand_lrd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_d = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut sequence = core::ExpansionSequence::new(instruction);
    super::shared::emit_ram_region_assertion(&mut sequence, rs1(instruction)?, allocator)?;
    sequence.emit_i_expanded(
        JoltInstructionKind::ADDI,
        v_reservation_d,
        rs1(instruction)?,
        0,
        allocator,
    )?;
    sequence.emit_i_expanded(
        JoltInstructionKind::ADDI,
        v_reservation_w,
        rs1(instruction)?,
        0,
        allocator,
    )?;
    sequence.emit_i_expanded(
        JoltInstructionKind::LD,
        rd(instruction)?,
        rs1(instruction)?,
        0,
        allocator,
    )?;
    sequence.finish()
}
