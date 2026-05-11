use super::*;

pub(in crate::expand) fn expand_scw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation = reservation_w_register();
    let v_reservation_d = reservation_d_register();
    let mut asm = ExpansionBuilder::new(*instruction);

    let ram_start = asm.allocate()?;
    super::shared::expand_ram_region_assertion(&mut asm, reg(rs1(instruction)?), ram_start)?;

    let v_success = asm.allocate()?;
    asm.dispatch_j(JoltInstructionKind::VirtualAdvice, v_success.operand(), 0);

    let v_one = asm.allocate()?;
    asm.dispatch_i(JoltInstructionKind::ADDI, v_one.operand(), reg(0), 1);
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertLTE,
        v_success.operand(),
        v_one.operand(),
        0,
    );
    asm.release(v_one);

    let v_addr_diff = asm.allocate()?;
    asm.dispatch_r(
        JoltInstructionKind::SUB,
        v_addr_diff.operand(),
        reg(v_reservation),
        reg(rs1(instruction)?),
    );
    asm.dispatch_r(
        JoltInstructionKind::MUL,
        v_addr_diff.operand(),
        v_success.operand(),
        v_addr_diff.operand(),
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertEQ,
        v_addr_diff.operand(),
        reg(0),
        0,
    );
    asm.release(v_addr_diff);

    asm.dispatch_i(
        JoltInstructionKind::ADDI,
        reg(v_reservation),
        v_success.operand(),
        0,
    );
    asm.release(v_success);

    let v_mem = asm.allocate()?;
    asm.dispatch_i(
        SourceInstructionKind::LW,
        v_mem.operand(),
        reg(rs1(instruction)?),
        0,
    );

    let v_diff = asm.allocate()?;
    asm.dispatch_r(
        JoltInstructionKind::SUB,
        v_diff.operand(),
        reg(rs2(instruction)?),
        v_mem.operand(),
    );
    asm.dispatch_r(
        JoltInstructionKind::MUL,
        v_diff.operand(),
        v_diff.operand(),
        reg(v_reservation),
    );
    asm.dispatch_r(
        JoltInstructionKind::ADD,
        v_diff.operand(),
        v_mem.operand(),
        v_diff.operand(),
    );
    asm.release(v_mem);

    asm.dispatch_i(
        JoltInstructionKind::ADDI,
        reg(v_reservation_d),
        v_diff.operand(),
        0,
    );
    asm.release(v_diff);
    asm.dispatch_s(
        SourceInstructionKind::SW,
        reg(rs1(instruction)?),
        reg(v_reservation_d),
        0,
    );
    asm.dispatch_i(
        JoltInstructionKind::XORI,
        reg(rd(instruction)?),
        reg(v_reservation),
        1,
    );
    asm.dispatch_i(JoltInstructionKind::ADDI, reg(v_reservation), reg(0), 0);
    asm.dispatch_i(JoltInstructionKind::ADDI, reg(v_reservation_d), reg(0), 0);

    asm.finalize()
}
