use super::*;

pub(in crate::expand) fn expand_scd(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation = reservation_d_register();
    let v_reservation_w = reservation_w_register();
    let mut asm = ExpansionBuilder::new(*instruction);

    let ram_start = asm.allocate()?;
    super::shared::expand_ram_region_assertion(&mut asm, reg(rs1(instruction)?), ram_start)?;

    let v_success = asm.allocate()?;
    asm.expand_j(SourceInstructionKind::VirtualAdvice, v_success.operand(), 0);

    let v_one = asm.allocate()?;
    asm.expand_i(SourceInstructionKind::ADDI, v_one.operand(), reg(0), 1);
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        v_success.operand(),
        v_one.operand(),
        0,
    );
    asm.release(v_one);

    let v_addr_diff = asm.allocate()?;
    asm.expand_r(
        SourceInstructionKind::SUB,
        v_addr_diff.operand(),
        reg(v_reservation),
        reg(rs1(instruction)?),
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        v_addr_diff.operand(),
        v_success.operand(),
        v_addr_diff.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertEQ,
        v_addr_diff.operand(),
        reg(0),
        0,
    );
    asm.release(v_addr_diff);

    let v_mem = asm.allocate()?;
    asm.expand_i(
        SourceInstructionKind::LD,
        v_mem.operand(),
        reg(rs1(instruction)?),
        0,
    );

    let v_diff = asm.allocate()?;
    asm.expand_r(
        SourceInstructionKind::SUB,
        v_diff.operand(),
        reg(rs2(instruction)?),
        v_mem.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        v_diff.operand(),
        v_diff.operand(),
        v_success.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::ADD,
        v_diff.operand(),
        v_mem.operand(),
        v_diff.operand(),
    );
    asm.release(v_mem);
    asm.expand_s(
        SourceInstructionKind::SD,
        reg(rs1(instruction)?),
        v_diff.operand(),
        0,
    );
    asm.release(v_diff);
    asm.expand_i(SourceInstructionKind::ADDI, reg(v_reservation), reg(0), 0);
    asm.expand_i(SourceInstructionKind::ADDI, reg(v_reservation_w), reg(0), 0);
    asm.expand_i(
        SourceInstructionKind::XORI,
        reg(rd(instruction)?),
        v_success.operand(),
        1,
    );
    asm.release(v_success);

    asm.finalize()
}
