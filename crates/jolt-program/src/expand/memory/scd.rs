use super::*;

/// Lowers `SC.D` to a conditional doubleword update driven by a success witness.
///
/// The tracer patches the first `VirtualAdvice` to `1` only when a doubleword
/// reservation covers `rs1`. The sequence proves that successful stores use the
/// reserved address, conditionally selects either `rs2` or the old memory
/// value, stores the selected doubleword, clears reservations, and returns
/// architectural status `0` on success or `1` on failure.
pub(in crate::expand) fn expand_scd(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation = reservation_d_register();
    let v_reservation_w = reservation_w_register();
    let mut asm = ExpansionBuilder::new(*instruction);

    let ram_start = asm.allocate()?;
    super::shared::expand_ram_region_assertion(&mut asm, reg(rs1(instruction)?), ram_start)?;

    let v_success = asm.allocate()?;
    // v_success is Boolean advice supplied by the tracer's LR/SC reservation
    // check: 1 means the SC succeeds, 0 means it fails.
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
    // If v_success is 1, the doubleword reservation address must equal rs1.
    // Failure leaves the address unconstrained because no write should occur.
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
    // v_diff = old_mem + success * (rs2 - old_mem), so failure stores the
    // previous memory value and success stores rs2.
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
    // RISC-V invalidates the reservation after every SC, regardless of success.
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
