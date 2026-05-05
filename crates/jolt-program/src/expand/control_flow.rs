use super::*;

pub(super) fn expand_csrrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = allocator
        .csr_to_virtual_register(csr)
        .ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    if rd(instruction)? == 0 {
        asm.emit_i(InstructionKind::ADDI, virtual_reg, rs1(instruction)?, 0)?;
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = asm.allocator().allocate()?;
        asm.emit_i(InstructionKind::ADDI, temp, rs1(instruction)?, 0)?;
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
        asm.emit_i(InstructionKind::ADDI, virtual_reg, temp, 0)?;
        asm.allocator().release(temp)?;
    } else {
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
        asm.emit_i(InstructionKind::ADDI, virtual_reg, rs1(instruction)?, 0)?;
    }
    asm.finalize()
}

pub(super) fn expand_csrrs(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = allocator
        .csr_to_virtual_register(csr)
        .ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    if rs1(instruction)? == 0 {
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
    } else if rd(instruction)? == 0 {
        asm.emit_r(
            InstructionKind::OR,
            virtual_reg,
            virtual_reg,
            rs1(instruction)?,
        )?;
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = asm.allocator().allocate()?;
        asm.emit_i(InstructionKind::ADDI, temp, rs1(instruction)?, 0)?;
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
        asm.emit_r(InstructionKind::OR, virtual_reg, virtual_reg, temp)?;
        asm.allocator().release(temp)?;
    } else {
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
        asm.emit_r(
            InstructionKind::OR,
            virtual_reg,
            virtual_reg,
            rs1(instruction)?,
        )?;
    }
    asm.finalize()
}

pub(super) fn expand_ebreak(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    let discard = asm.allocator().allocate()?;
    asm.emit_j(InstructionKind::JAL, discard, 0)?;
    asm.allocator().release(discard)?;
    asm.finalize()
}

pub(super) fn expand_ecall(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    const MCAUSE_ECALL_FROM_MMODE: i128 = 11;

    let v_trap_handler_reg = allocator.trap_handler_register();
    let vr_mepc = allocator.mepc_register();
    let vr_mcause = allocator.mcause_register();
    let vr_mtval = allocator.mtval_register();
    let vr_mstatus = allocator.mstatus_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    let ecall_addr = asm.allocator().allocate()?;
    asm.emit_u(InstructionKind::AUIPC, ecall_addr, 0)?;
    asm.emit_i(InstructionKind::ADDI, vr_mepc, ecall_addr, 0)?;
    asm.allocator().release(ecall_addr)?;

    asm.emit_i(InstructionKind::ADDI, vr_mcause, 0, MCAUSE_ECALL_FROM_MMODE)?;
    asm.emit_i(InstructionKind::ADDI, vr_mtval, 0, 0)?;

    let three = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::ADDI, three, 0, 3)?;
    asm.emit_i(InstructionKind::SLLI, vr_mstatus, three, 11)?;
    asm.allocator().release(three)?;

    let jalr_rd = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::JALR, jalr_rd, v_trap_handler_reg, 0)?;
    asm.allocator().release(jalr_rd)?;

    asm.finalize()
}

pub(super) fn expand_mret(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mepc_vr = allocator.mepc_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    let jalr_rd = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::JALR, jalr_rd, mepc_vr, 0)?;
    asm.allocator().release(jalr_rd)?;
    asm.finalize()
}
