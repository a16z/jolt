use common::constants::RAM_START_ADDRESS;

use super::*;
use crate::jolt_asm;

/// Emits the common LR/SC proof guard that rejects non-RAM reservation targets.
///
/// Jolt models LR/SC reservations only for ordinary RAM. This assertion keeps
/// synthesized failure-path stores from touching memory-mapped I/O addresses.
pub(in crate::expand) fn expand_ram_region_assertion(
    asm: &mut ExpansionBuilder,
    address_register: RegisterOperand,
    ram_start: TempId,
) -> Result<(), ExpansionError> {
    asm.emit_u(
        SourceInstructionKind::LUI,
        ram_start.operand(),
        RAM_START_ADDRESS as i128,
    );
    asm.emit_b(
        SourceInstructionKind::VirtualAssertLTE,
        ram_start.operand(),
        address_register,
        0,
    );
    asm.release(ram_start);
    Ok(())
}

/// Lowers `LB`/`LBU` by loading the containing doubleword and extracting a byte.
///
/// The effective address is rounded down to the aligned 8-byte address for the
/// `LD`. `VirtualWindowMaskB` turns the effective address into the byte mask of
/// the addressed lane, and a single fused parallel-extract lookup (signed or
/// unsigned) pulls the byte out of the loaded doubleword.
pub(in crate::expand) fn expand_byte_load(
    instruction: &SourceInstructionRow,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let base = reg(rs1(instruction)?);
    let destination = reg(rd(instruction)?);
    let offset = format_i_imm(instruction.operands.imm);

    // v1 = aligned address of the containing doubleword; the fused lookup
    // computes `(rs1 + imm) & !7` in one row.
    jolt_asm!(asm, {
        align_addr v1, base, offset;
        ld v1, v1, 0;
        window_mask_b v0, base, offset;
    });
    if signed {
        jolt_asm!(asm, { pext_signed destination, v1, v0; });
    } else {
        jolt_asm!(asm, { pext destination, v1, v0; });
    }
    asm.release_many([v0, v1]);

    asm.finalize()
}

/// Lowers `LH`/`LHU` by loading the containing doubleword and extracting a halfword.
///
/// Halfword alignment is asserted first. The extraction mirrors byte loads:
/// `VirtualWindowMaskH` builds the halfword lane's byte mask from the
/// effective address, and a fused parallel-extract lookup (signed or unsigned)
/// pulls the lane out of the loaded doubleword.
pub(in crate::expand) fn expand_halfword_load(
    instruction: &SourceInstructionRow,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let base = reg(rs1(instruction)?);
    let destination = reg(rd(instruction)?);
    let offset = instruction.operands.imm;
    let formatted_offset = format_i_imm(offset);

    // Halfword loads may start at byte offsets 0, 2, 4, or 6 within the
    // containing doubleword.
    // v1 = aligned address of the containing doubleword; the fused lookup
    // computes `(rs1 + imm) & !7` in one row.
    jolt_asm!(asm, {
        assert_halfword_alignment base, offset;
        align_addr v1, base, formatted_offset;
        ld v1, v1, 0;
        window_mask_h v0, base, formatted_offset;
    });
    if signed {
        jolt_asm!(asm, { pext_signed destination, v1, v0; });
    } else {
        jolt_asm!(asm, { pext destination, v1, v0; });
    }
    asm.release_many([v0, v1]);

    asm.finalize()
}

/// Lowers `LW`/`LWU` by loading the containing doubleword and extracting a word.
///
/// The word alignment assertion is required by the source semantics; it also
/// guarantees the effective address's bits 0-1 are zero, which
/// `VirtualWindowMaskW` relies on (it reads only bit 2). A fused
/// parallel-extract lookup (signed or unsigned) pulls the word lane out of the
/// loaded doubleword.
pub(in crate::expand) fn expand_word_load(
    instruction: &SourceInstructionRow,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let base = reg(rs1(instruction)?);
    let destination = reg(rd(instruction)?);
    let offset = instruction.operands.imm;
    let formatted_offset = format_i_imm(offset);

    // v1 = aligned address of the containing doubleword; the fused lookup
    // computes `(rs1 + imm) & !7` in one row.
    jolt_asm!(asm, {
        assert_word_alignment base, offset;
        align_addr v1, base, formatted_offset;
        ld v1, v1, 0;
        window_mask_w v0, base, formatted_offset;
    });
    if signed {
        jolt_asm!(asm, { pext_signed destination, v1, v0; });
    } else {
        jolt_asm!(asm, { pext destination, v1, v0; });
    }
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}

/// Lowers an advice load with byte length 1, 2, 4, or 8.
///
/// `VirtualAdviceLoad` reads from the advice tape rather than RAM. Narrow
/// advice loads are signed loads, so the helper left-shifts the value into the
/// high bits and arithmetic-shifts it back to XLEN.
pub(in crate::expand) fn expand_advice_load(
    instruction: &SourceInstructionRow,
    byte_len: i128,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_j(
        SourceInstructionKind::VirtualAdviceLoad(jolt_riscv::instructions::VirtualAdviceLoad(())),
        reg(rd(instruction)?),
        byte_len,
    );
    if byte_len < 8 {
        let shift = 64 - byte_len * 8;
        asm.emit_i(
            SourceInstructionKind::SLLI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
        asm.emit_i(
            SourceInstructionKind::SRAI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
    }

    asm.finalize()
}

/// Lowers arithmetic/bitwise doubleword AMOs using the shared read-modify-write shape.
///
/// The old memory value is loaded, `op(old, rs2)` is stored back, and the old
/// value is copied to `rd`. Jolt traces are sequential, so the atomicity
/// contract reduces to preserving this single-instruction read/modify/write
/// order in the expanded bytecode.
pub(in crate::expand) fn expand_amo_d(
    instruction: &SourceInstructionRow,
    op: SourceInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs2 = asm.allocate()?;
    let v_rd = asm.allocate()?;

    asm.emit_i(
        SourceInstructionKind::LD,
        v_rd.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_r(op, v_rs2.operand(), v_rd.operand(), reg(rs2(instruction)?));
    asm.emit_s(
        SourceInstructionKind::SD,
        reg(rs1(instruction)?),
        v_rs2.operand(),
        0,
    );
    asm.emit_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v_rd.operand(),
        0,
    );
    asm.release_many([v_rs2, v_rd]);

    asm.finalize()
}

/// Lowers signed or unsigned doubleword AMO min/max.
///
/// `compare_op` decides whether `rs2` should replace the old memory value.
/// The update is computed as `old + take_rs2 * (rs2 - old)`, so the same
/// arithmetic shape handles min and max once the comparison operands are
/// ordered appropriately.
pub(in crate::expand) fn expand_amo_minmax_d(
    instruction: &SourceInstructionRow,
    compare_op: SourceInstructionKind,
    min: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let (cmp_rs1, cmp_rs2): (RegisterOperand, RegisterOperand) = if min {
        (reg(rs2(instruction)?), v0.operand())
    } else {
        (v0.operand(), reg(rs2(instruction)?))
    };

    // v0 = old memory. v1 = whether rs2 should be stored. v2 = conditional
    // delta from old memory to rs2.
    asm.emit_i(
        SourceInstructionKind::LD,
        v0.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_r(compare_op, v1.operand(), cmp_rs1, cmp_rs2);
    asm.emit_r(
        SourceInstructionKind::SUB,
        v2.operand(),
        reg(rs2(instruction)?),
        v0.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::MUL,
        v2.operand(),
        v2.operand(),
        v1.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::ADD,
        v1.operand(),
        v0.operand(),
        v2.operand(),
    );
    asm.emit_s(
        SourceInstructionKind::SD,
        reg(rs1(instruction)?),
        v1.operand(),
        0,
    );
    asm.emit_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release_many([v0, v1, v2]);

    asm.finalize()
}

/// Lowers arithmetic/bitwise word AMOs by updating one word lane in a doubleword.
///
/// The pre-helper extracts the old word from the containing doubleword. The
/// final operation is performed on that word, and the post-helper merges the
/// low word of the result back into the containing doubleword while returning
/// the old word sign-extended in `rd`.
pub(in crate::expand) fn expand_amo_w(
    instruction: &SourceInstructionRow,
    op: SourceInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;
    let v_rs2 = asm.allocate()?;
    let v_mask = asm.allocate()?;
    let v_dword = asm.allocate()?;
    let v_shift = asm.allocate()?;

    expand_amo_pre64(
        &mut asm,
        reg(rs1(instruction)?),
        v_rd.operand(),
        v_dword.operand(),
        v_shift.operand(),
    )?;
    asm.emit_r(op, v_rs2.operand(), v_rd.operand(), reg(rs2(instruction)?));
    expand_amo_post64(
        &mut asm,
        AmoPost64 {
            rs1: reg(rs1(instruction)?),
            v_rs2: v_rs2.operand(),
            v_dword: v_dword.operand(),
            v_shift: v_shift.operand(),
            v_mask: v_mask.operand(),
            rd: reg(rd(instruction)?),
            v_rd: v_rd.operand(),
        },
    )?;
    asm.release_many([v_rd, v_rs2, v_mask, v_dword, v_shift]);

    asm.finalize()
}

/// Lowers signed or unsigned word AMO min/max.
///
/// The old word and `rs2` are extended according to the comparison mode before
/// `compare_op` runs. The stored value is still the selected low word, merged
/// back into the containing doubleword by `expand_amo_post64`.
pub(in crate::expand) fn expand_amo_minmax_w(
    instruction: &SourceInstructionRow,
    compare_op: SourceInstructionKind,
    min: bool,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;
    let v_dword = asm.allocate()?;
    let v_shift = asm.allocate()?;

    expand_amo_pre64(
        &mut asm,
        reg(rs1(instruction)?),
        v_rd.operand(),
        v_dword.operand(),
        v_shift.operand(),
    )?;

    let v_rs2 = asm.allocate()?;
    let v0 = asm.allocate()?;
    let extend_op = if signed {
        SourceInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        )
    } else {
        SourceInstructionKind::VirtualZeroExtendWord(
            jolt_riscv::instructions::VirtualZeroExtendWord(()),
        )
    };
    // Compare normalized word values, but keep the original low-word payload
    // for the value that will be merged back into memory.
    asm.emit_i(extend_op, v_rs2.operand(), reg(rs2(instruction)?), 0);
    asm.emit_i(extend_op, v0.operand(), v_rd.operand(), 0);
    let (cmp_rs1, cmp_rs2) = if min {
        (v_rs2.operand(), v0.operand())
    } else {
        (v0.operand(), v_rs2.operand())
    };
    asm.emit_r(compare_op, v0.operand(), cmp_rs1, cmp_rs2);
    asm.emit_r(
        SourceInstructionKind::SUB,
        v_rs2.operand(),
        reg(rs2(instruction)?),
        v_rd.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::MUL,
        v_rs2.operand(),
        v_rs2.operand(),
        v0.operand(),
    );
    asm.emit_r(
        SourceInstructionKind::ADD,
        v_rs2.operand(),
        v_rs2.operand(),
        v_rd.operand(),
    );
    expand_amo_post64(
        &mut asm,
        AmoPost64 {
            rs1: reg(rs1(instruction)?),
            v_rs2: v_rs2.operand(),
            v_dword: v_dword.operand(),
            v_shift: v_shift.operand(),
            v_mask: v0.operand(),
            rd: reg(rd(instruction)?),
            v_rd: v_rd.operand(),
        },
    )?;
    asm.release_many([v_rd, v_dword, v_shift, v_rs2, v0]);

    asm.finalize()
}

/// Reads the containing doubleword and extracts the selected word for word AMOs.
///
/// `rs1` must be word-aligned. `v_dword` receives the containing aligned
/// doubleword, `v_shift` receives the byte offset times eight, and `v_rd`
/// receives the selected old word in its low 32 bits.
pub(in crate::expand) fn expand_amo_pre64(
    asm: &mut ExpansionBuilder,
    rs1: RegisterOperand,
    v_rd: RegisterOperand,
    v_dword: RegisterOperand,
    v_shift: RegisterOperand,
) -> Result<(), ExpansionError> {
    asm.emit_address(SourceInstructionKind::VirtualAssertWordAlignment, rs1, 0);
    asm.emit_i(SourceInstructionKind::ANDI, v_shift, rs1, format_i_imm(-8));
    asm.emit_i(SourceInstructionKind::LD, v_dword, v_shift, 0);
    asm.emit_i(SourceInstructionKind::SLLI, v_shift, rs1, 3);
    asm.emit_r(SourceInstructionKind::SRL, v_rd, v_dword, v_shift);
    Ok(())
}

/// Register bundle consumed by `expand_amo_post64`.
///
/// The post-helper needs both the containing doubleword state and the selected
/// lane metadata from `expand_amo_pre64`, plus the new word value and the
/// architectural destination for the old word.
pub(in crate::expand) struct AmoPost64 {
    pub(in crate::expand) rs1: RegisterOperand,
    pub(in crate::expand) v_rs2: RegisterOperand,
    pub(in crate::expand) v_dword: RegisterOperand,
    pub(in crate::expand) v_shift: RegisterOperand,
    pub(in crate::expand) v_mask: RegisterOperand,
    pub(in crate::expand) rd: RegisterOperand,
    pub(in crate::expand) v_rd: RegisterOperand,
}

/// Merges a word-AMO result into its containing doubleword and returns old word.
///
/// `v_rs2` is shifted into the selected lane, XORed with the old doubleword,
/// masked to that lane, and XORed back. This updates only the selected 32 bits
/// before storing the containing doubleword and sign-extending the old word to
/// `rd`.
pub(in crate::expand) fn expand_amo_post64(
    asm: &mut ExpansionBuilder,
    registers: AmoPost64,
) -> Result<(), ExpansionError> {
    let AmoPost64 {
        rs1,
        v_rs2,
        v_dword,
        v_shift,
        v_mask,
        rd,
        v_rd,
    } = registers;

    // Build a 32-bit lane mask, shift the new word into place, and use
    // masked-XOR replacement: new_dword = old ^ ((old ^ new) & mask).
    asm.emit_i(SourceInstructionKind::ORI, v_mask, reg(0), format_i_imm(-1));
    asm.emit_i(SourceInstructionKind::SRLI, v_mask, v_mask, 32);
    asm.emit_r(SourceInstructionKind::SLL, v_mask, v_mask, v_shift);
    asm.emit_r(SourceInstructionKind::SLL, v_shift, v_rs2, v_shift);
    asm.emit_r(SourceInstructionKind::XOR, v_shift, v_dword, v_shift);
    asm.emit_r(SourceInstructionKind::AND, v_shift, v_shift, v_mask);
    asm.emit_r(SourceInstructionKind::XOR, v_dword, v_dword, v_shift);
    asm.emit_i(SourceInstructionKind::ANDI, v_mask, rs1, format_i_imm(-8));
    asm.emit_s(SourceInstructionKind::SD, v_mask, v_dword, 0);
    asm.emit_i(
        SourceInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        ),
        rd,
        v_rd,
        0,
    );
    Ok(())
}

/// Lowers a narrow store (`SB`/`SH`/`SW`) via a fused read-modify-write.
///
/// The containing doubleword is loaded, the addressed lane is cleared with the
/// window mask (`ANDN`) and replaced by the store data shifted into position
/// (`ShiftData`); the lane bits are disjoint from the cleared doubleword, so a
/// plain `ADD` merges them before the `SD` writes the doubleword back.
pub(in crate::expand) fn expand_narrow_store(
    instruction: &SourceInstructionRow,
    window_mask: SourceInstructionKind,
    shift_data: SourceInstructionKind,
    alignment: Option<SourceInstructionKind>,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;
    let base = reg(rs1(instruction)?);
    let source = reg(rs2(instruction)?);
    let offset = instruction.operands.imm;
    let formatted_offset = format_i_imm(offset);

    if let Some(alignment) = alignment {
        // `SH`/`SW` assert alignment; `SB` passes `None`. The asserts also
        // guarantee the offset bits the window-mask and shift-data tables do
        // not read are zero.
        asm.emit_address(alignment, base, offset);
    }
    jolt_asm!(asm, {
        addi v0, base, formatted_offset;
        andi v1, v0, format_i_imm(-8);
        ld v2, v1, 0;
    });
    // v3 = mask of the addressed lane.
    asm.emit_i(window_mask, v3.operand(), v0.operand(), 0);
    jolt_asm!(asm, { andn v2, v2, v3; });
    // v3 = store data shifted into the lane.
    asm.emit_r(shift_data, v3.operand(), source, v0.operand());
    jolt_asm!(asm, {
        add v2, v2, v3;
        sd v1, v2, 0;
    });
    asm.release_many([v0, v1, v2, v3]);

    asm.finalize()
}
