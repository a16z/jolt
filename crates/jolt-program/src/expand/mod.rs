//! RV64 bytecode expansion from decoded source rows into final Jolt bytecode.
//!
//! Expansion intentionally has no `Xlen` parameter: the `jolt-program` pipeline
//! only supports RV64. RV32/ELF32 inputs should be rejected before this module is
//! called.

pub mod allocator;
pub mod assembler;
pub mod error;
pub mod metadata;
pub mod sequences;

pub use allocator::ExpansionAllocator;
pub use error::ExpansionError;

use jolt_riscv::{InstructionKind, NormalizedInstruction, NormalizedOperands};

pub trait InlineExpansionProvider {
    fn expand_inline(
        &mut self,
        instruction: &NormalizedInstruction,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError>;
}

#[derive(Debug, Default)]
pub struct NoInlineExpansionProvider;

impl InlineExpansionProvider for NoInlineExpansionProvider {
    fn expand_inline(
        &mut self,
        _instruction: &NormalizedInstruction,
        _allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        Err(ExpansionError::InlineProviderRequired)
    }
}

pub fn expand_instruction(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_instruction_with_provider(instruction, allocator, &mut NoInlineExpansionProvider)
}

pub fn expand_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    if instruction.operands.rd == Some(0)
        && !handles_rd_zero_internally(instruction.instruction_kind)
    {
        if has_side_effects(instruction.instruction_kind) {
            let virtual_register = allocator.allocate()?;
            let mut rewritten = *instruction;
            rewritten.operands.rd = Some(virtual_register);
            let expanded = expand_instruction_with_provider(&rewritten, allocator, inline_provider);
            allocator.release(virtual_register)?;
            return expanded;
        }
        return Ok(vec![noop_for(*instruction)]);
    }

    match instruction.instruction_kind {
        InstructionKind::Inline => inline_provider.expand_inline(instruction, allocator),
        InstructionKind::ADDIW => expand_addiw(instruction, allocator),
        InstructionKind::ADDW => expand_addw(instruction, allocator),
        InstructionKind::SUBW => expand_subw(instruction, allocator),
        InstructionKind::MULH => expand_mulh(instruction, allocator),
        InstructionKind::MULHSU => expand_mulhsu(instruction, allocator),
        InstructionKind::MULW => expand_mulw(instruction, allocator),
        InstructionKind::LB => expand_lb(instruction, allocator),
        InstructionKind::LBU => expand_lbu(instruction, allocator),
        InstructionKind::LH => expand_lh(instruction, allocator),
        InstructionKind::LHU => expand_lhu(instruction, allocator),
        InstructionKind::LW => expand_lw(instruction, allocator),
        InstructionKind::LWU => expand_lwu(instruction, allocator),
        InstructionKind::AdviceLB => expand_advice_load(instruction, 1, Some(56), allocator),
        InstructionKind::AdviceLH => expand_advice_load(instruction, 2, Some(48), allocator),
        InstructionKind::AdviceLW => expand_advice_load(instruction, 4, Some(32), allocator),
        InstructionKind::AdviceLD => expand_advice_load(instruction, 8, None, allocator),
        InstructionKind::AMOADDD => expand_amo_d(instruction, InstructionKind::ADD, allocator),
        InstructionKind::AMOANDD => expand_amo_d(instruction, InstructionKind::AND, allocator),
        InstructionKind::AMOORD => expand_amo_d(instruction, InstructionKind::OR, allocator),
        InstructionKind::AMOXORD => expand_amo_d(instruction, InstructionKind::XOR, allocator),
        InstructionKind::AMOSWAPD => expand_amoswapd(instruction, allocator),
        InstructionKind::AMOMAXD => {
            expand_amo_minmax_d(instruction, InstructionKind::SLT, false, allocator)
        }
        InstructionKind::AMOMAXUD => {
            expand_amo_minmax_d(instruction, InstructionKind::SLTU, false, allocator)
        }
        InstructionKind::AMOMIND => {
            expand_amo_minmax_d(instruction, InstructionKind::SLT, true, allocator)
        }
        InstructionKind::AMOMINUD => {
            expand_amo_minmax_d(instruction, InstructionKind::SLTU, true, allocator)
        }
        InstructionKind::AMOADDW => expand_amo_w(instruction, InstructionKind::ADD, allocator),
        InstructionKind::AMOANDW => expand_amo_w(instruction, InstructionKind::AND, allocator),
        InstructionKind::AMOORW => expand_amo_w(instruction, InstructionKind::OR, allocator),
        InstructionKind::AMOXORW => expand_amo_w(instruction, InstructionKind::XOR, allocator),
        InstructionKind::AMOSWAPW => expand_amoswapw(instruction, allocator),
        InstructionKind::AMOMAXW => {
            expand_amo_minmax_w(instruction, InstructionKind::SLT, false, true, allocator)
        }
        InstructionKind::AMOMAXUW => {
            expand_amo_minmax_w(instruction, InstructionKind::SLTU, false, false, allocator)
        }
        InstructionKind::AMOMINW => {
            expand_amo_minmax_w(instruction, InstructionKind::SLT, true, true, allocator)
        }
        InstructionKind::AMOMINUW => {
            expand_amo_minmax_w(instruction, InstructionKind::SLTU, true, false, allocator)
        }
        InstructionKind::LRD => expand_lrd(instruction, allocator),
        InstructionKind::LRW => expand_lrw(instruction, allocator),
        InstructionKind::DIV => expand_div(instruction, allocator),
        InstructionKind::DIVU => expand_divu(instruction, allocator),
        InstructionKind::DIVW => expand_divw(instruction, allocator),
        InstructionKind::DIVUW => expand_divuw(instruction, allocator),
        InstructionKind::REM => expand_rem(instruction, allocator),
        InstructionKind::REMU => expand_remu(instruction, allocator),
        InstructionKind::REMW => expand_remw(instruction, allocator),
        InstructionKind::REMUW => expand_remuw(instruction, allocator),
        InstructionKind::SB => expand_sb(instruction, allocator),
        InstructionKind::SCD => expand_scd(instruction, allocator),
        InstructionKind::SCW => expand_scw(instruction, allocator),
        InstructionKind::SH => expand_sh(instruction, allocator),
        InstructionKind::SW => expand_sw(instruction, allocator),
        InstructionKind::CSRRW => expand_csrrw(instruction, allocator),
        InstructionKind::CSRRS => expand_csrrs(instruction, allocator),
        InstructionKind::EBREAK => expand_ebreak(instruction, allocator),
        InstructionKind::ECALL => expand_ecall(instruction, allocator),
        InstructionKind::MRET => expand_mret(instruction, allocator),
        InstructionKind::SLL => expand_sll(instruction, allocator),
        InstructionKind::SLLI => expand_slli(instruction, allocator),
        InstructionKind::SLLW => expand_sllw(instruction, allocator),
        InstructionKind::SLLIW => expand_slliw(instruction, allocator),
        InstructionKind::SRL => expand_srl(instruction, allocator),
        InstructionKind::SRLI => expand_srli(instruction, allocator),
        InstructionKind::SRA => expand_sra(instruction, allocator),
        InstructionKind::SRAI => expand_srai(instruction, allocator),
        InstructionKind::SRLIW => expand_srliw(instruction, allocator),
        InstructionKind::SRAIW => expand_sraiw(instruction, allocator),
        InstructionKind::SRLW => expand_srlw(instruction, allocator),
        InstructionKind::SRAW => expand_sraw(instruction, allocator),
        _ => Ok(vec![*instruction]),
    }
}

pub fn expand_program(
    instructions: impl IntoIterator<Item = NormalizedInstruction>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_program_with_provider(instructions, &mut NoInlineExpansionProvider)
}

pub fn expand_program_with_provider<P: InlineExpansionProvider + ?Sized>(
    instructions: impl IntoIterator<Item = NormalizedInstruction>,
    inline_provider: &mut P,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let mut expanded = Vec::new();
    for instruction in instructions {
        expanded.extend(expand_instruction_with_provider(
            &instruction,
            &mut allocator,
            inline_provider,
        )?);
    }
    Ok(expanded)
}

fn noop_for(instruction: NormalizedInstruction) -> NormalizedInstruction {
    NormalizedInstruction {
        instruction_kind: InstructionKind::ADDI,
        address: instruction.address,
        operands: NormalizedOperands {
            rd: Some(0),
            rs1: Some(0),
            rs2: None,
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: instruction.is_compressed,
    }
}

fn expand_addiw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::ADDI,
        rd(instruction)?,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    asm.finalize()
}

fn expand_addw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_r(
        InstructionKind::ADD,
        rd(instruction)?,
        rs1(instruction)?,
        rs2(instruction)?,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    asm.finalize()
}

fn expand_subw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_r(
        InstructionKind::SUB,
        rd(instruction)?,
        rs1(instruction)?,
        rs2(instruction)?,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    asm.finalize()
}

fn expand_mulw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_r(
        InstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        rs2(instruction)?,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    asm.finalize()
}

fn expand_mulh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_sx = allocator.allocate()?;
    let v_sy = allocator.allocate()?;
    let v_tmp = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualMovsign, v_sx, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::VirtualMovsign, v_sy, rs2(instruction)?, 0)?;
    asm.emit_r(InstructionKind::MUL, v_sx, v_sx, rs2(instruction)?)?;
    asm.emit_r(InstructionKind::MUL, v_sy, v_sy, rs1(instruction)?)?;
    asm.emit_r(
        InstructionKind::MULHU,
        v_tmp,
        rs1(instruction)?,
        rs2(instruction)?,
    )?;
    asm.emit_r(InstructionKind::ADD, v_tmp, v_tmp, v_sx)?;
    asm.emit_r(InstructionKind::ADD, rd(instruction)?, v_tmp, v_sy)?;
    let sequence = asm.finalize()?;
    allocator.release(v_sx)?;
    allocator.release(v_sy)?;
    allocator.release(v_tmp)?;
    Ok(sequence)
}

fn expand_mulhsu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualMovsign, v0, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, 1)?;
    asm.emit_r(InstructionKind::XOR, v2, rs1(instruction)?, v0)?;
    asm.emit_r(InstructionKind::ADD, v2, v2, v1)?;
    asm.emit_r(InstructionKind::MULHU, v3, v2, rs2(instruction)?)?;
    asm.emit_r(InstructionKind::MUL, v2, v2, rs2(instruction)?)?;
    asm.emit_r(InstructionKind::XOR, v3, v3, v0)?;
    asm.emit_r(InstructionKind::XOR, v2, v2, v0)?;
    asm.emit_r(InstructionKind::ADD, v0, v2, v1)?;
    asm.emit_r(InstructionKind::SLTU, v0, v0, v2)?;
    asm.emit_r(InstructionKind::ADD, rd(instruction)?, v3, v0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    allocator.release(v3)?;
    Ok(sequence)
}

fn expand_lb(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_byte_load(instruction, allocator, true)
}

fn expand_lbu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_byte_load(instruction, allocator, false)
}

fn expand_byte_load(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    signed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::XORI, v0, v0, 7)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SLL, v1, v1, v0)?;
    asm.emit_i(
        if signed {
            InstructionKind::SRAI
        } else {
            InstructionKind::SRLI
        },
        rd(instruction)?,
        v1,
        56,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

fn expand_lh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_halfword_load(instruction, allocator, true)
}

fn expand_lhu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_halfword_load(instruction, allocator, false)
}

fn expand_halfword_load(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    signed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertHalfwordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::XORI, v0, v0, 6)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SLL, v1, v1, v0)?;
    asm.emit_i(
        if signed {
            InstructionKind::SRAI
        } else {
            InstructionKind::SRLI
        },
        rd(instruction)?,
        v1,
        48,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

fn expand_lw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SRL, v1, v1, v0)?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        v1,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

fn expand_lwu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::XORI, v0, v0, 4)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SLL, v1, v1, v0)?;
    asm.emit_i(InstructionKind::SRLI, rd(instruction)?, v1, 32)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

fn expand_lrd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_d = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::LD, rd(instruction)?, rs1(instruction)?, 0)?;
    asm.finalize()
}

fn expand_lrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_w = allocator.reservation_w_register();
    let v_reservation_d = allocator.reservation_d_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, 0, 0)?;
    asm.emit_i(InstructionKind::LW, rd(instruction)?, rs1(instruction)?, 0)?;
    asm.finalize()
}

fn expand_advice_load(
    instruction: &NormalizedInstruction,
    byte_len: i128,
    sign_extension_shift: Option<i128>,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_j(
        InstructionKind::VirtualAdviceLoad,
        rd(instruction)?,
        byte_len,
    )?;
    if let Some(shift) = sign_extension_shift {
        asm.emit_i(
            InstructionKind::SLLI,
            rd(instruction)?,
            rd(instruction)?,
            shift,
        )?;
        asm.emit_i(
            InstructionKind::SRAI,
            rd(instruction)?,
            rd(instruction)?,
            shift,
        )?;
    }
    asm.finalize()
}

fn expand_amo_d(
    instruction: &NormalizedInstruction,
    op: InstructionKind,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs2 = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.emit_r(op, v_rs2, v_rd, rs2(instruction)?)?;
    asm.emit_s(InstructionKind::SD, rs1(instruction)?, v_rs2, 0)?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs2)?;
    allocator.release(v_rd)?;
    Ok(sequence)
}

fn expand_amoswapd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.emit_s(InstructionKind::SD, rs1(instruction)?, rs2(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v_rd)?;
    Ok(sequence)
}

fn expand_amo_minmax_d(
    instruction: &NormalizedInstruction,
    compare_op: InstructionKind,
    min: bool,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::LD, v0, rs1(instruction)?, 0)?;
    let (cmp_rs1, cmp_rs2) = if min {
        (rs2(instruction)?, v0)
    } else {
        (v0, rs2(instruction)?)
    };
    asm.emit_r(compare_op, v1, cmp_rs1, cmp_rs2)?;
    asm.emit_r(InstructionKind::SUB, v2, rs2(instruction)?, v0)?;
    asm.emit_r(InstructionKind::MUL, v2, v2, v1)?;
    asm.emit_r(InstructionKind::ADD, v1, v0, v2)?;
    asm.emit_s(InstructionKind::SD, rs1(instruction)?, v1, 0)?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    Ok(sequence)
}

fn expand_amo_w(
    instruction: &NormalizedInstruction,
    op: InstructionKind,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let v_rs2 = allocator.allocate()?;
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;
    asm.emit_r(op, v_rs2, v_rd, rs2(instruction)?)?;
    amo_post64(
        &mut asm,
        rs1(instruction)?,
        v_rs2,
        v_dword,
        v_shift,
        v_mask,
        rd(instruction)?,
        v_rd,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rd)?;
    allocator.release(v_rs2)?;
    allocator.release(v_mask)?;
    allocator.release(v_dword)?;
    allocator.release(v_shift)?;
    Ok(sequence)
}

fn expand_amoswapw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;
    amo_post64(
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

fn expand_amo_minmax_w(
    instruction: &NormalizedInstruction,
    compare_op: InstructionKind,
    min: bool,
    signed: bool,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;

    let v_rs2 = asm.allocator().allocate()?;
    let v0 = asm.allocator().allocate()?;
    let extend_op = if signed {
        InstructionKind::VirtualSignExtendWord
    } else {
        InstructionKind::VirtualZeroExtendWord
    };
    asm.emit_i(extend_op, v_rs2, rs2(instruction)?, 0)?;
    asm.emit_i(extend_op, v0, v_rd, 0)?;
    let (cmp_rs1, cmp_rs2) = if min { (v_rs2, v0) } else { (v0, v_rs2) };
    asm.emit_r(compare_op, v0, cmp_rs1, cmp_rs2)?;
    asm.emit_r(InstructionKind::SUB, v_rs2, rs2(instruction)?, v_rd)?;
    asm.emit_r(InstructionKind::MUL, v_rs2, v_rs2, v0)?;
    asm.emit_r(InstructionKind::ADD, v_rs2, v_rs2, v_rd)?;
    amo_post64(
        &mut asm,
        rs1(instruction)?,
        v_rs2,
        v_dword,
        v_shift,
        v0,
        rd(instruction)?,
        v_rd,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rd)?;
    allocator.release(v_dword)?;
    allocator.release(v_shift)?;
    allocator.release(v_rs2)?;
    allocator.release(v0)?;
    Ok(sequence)
}

fn amo_pre64(
    asm: &mut assembler::InstrAssembler<'_>,
    rs1: u8,
    v_rd: u8,
    v_dword: u8,
    v_shift: u8,
) -> Result<(), ExpansionError> {
    asm.emit_align(InstructionKind::VirtualAssertWordAlignment, rs1, 0)?;
    asm.emit_i(InstructionKind::ANDI, v_shift, rs1, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v_dword, v_shift, 0)?;
    asm.emit_i(InstructionKind::SLLI, v_shift, rs1, 3)?;
    asm.emit_r(InstructionKind::SRL, v_rd, v_dword, v_shift)?;
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn amo_post64(
    asm: &mut assembler::InstrAssembler<'_>,
    rs1: u8,
    v_rs2: u8,
    v_dword: u8,
    v_shift: u8,
    v_mask: u8,
    rd: u8,
    v_rd: u8,
) -> Result<(), ExpansionError> {
    asm.emit_i(InstructionKind::ORI, v_mask, 0, format_i_imm(-1))?;
    asm.emit_i(InstructionKind::SRLI, v_mask, v_mask, 32)?;
    asm.emit_r(InstructionKind::SLL, v_mask, v_mask, v_shift)?;
    asm.emit_r(InstructionKind::SLL, v_shift, v_rs2, v_shift)?;
    asm.emit_r(InstructionKind::XOR, v_shift, v_dword, v_shift)?;
    asm.emit_r(InstructionKind::AND, v_shift, v_shift, v_mask)?;
    asm.emit_r(InstructionKind::XOR, v_dword, v_dword, v_shift)?;
    asm.emit_i(InstructionKind::ANDI, v_mask, rs1, format_i_imm(-8))?;
    asm.emit_s(InstructionKind::SD, v_mask, v_dword, 0)?;
    asm.emit_i(InstructionKind::VirtualSignExtendWord, rd, v_rd, 0)?;
    Ok(())
}

fn expand_div(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_signed_div_rem(instruction, allocator, false, false)
}

fn expand_rem(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_signed_div_rem(instruction, allocator, false, true)
}

fn expand_signed_div_rem(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    word: bool,
    remainder_output: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let a0 = rs1(instruction)?;
    let a1 = rs2(instruction)?;
    let a2 = allocator.allocate()?;
    let a3 = allocator.allocate()?;
    let t0 = allocator.allocate()?;
    let t1 = allocator.allocate()?;
    let (mut t2, mut t3, t4) = if word {
        (
            allocator.allocate()?,
            allocator.allocate()?,
            Some(allocator.allocate()?),
        )
    } else {
        (0, 0, None)
    };
    let dividend = t4.unwrap_or(a0);
    let divisor = if word { t3 } else { a1 };
    let shmat = if word { 31 } else { 63 };
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    asm.emit_j(InstructionKind::VirtualAdvice, a2, 0)?;
    asm.emit_j(InstructionKind::VirtualAdvice, a3, 0)?;

    if word {
        asm.emit_i(InstructionKind::VirtualSignExtendWord, dividend, a0, 0)?;
        asm.emit_i(InstructionKind::VirtualSignExtendWord, divisor, a1, 0)?;
    }

    asm.emit_b(InstructionKind::VirtualAssertValidDiv0, divisor, a2, 0)?;
    asm.emit_r(
        if word {
            InstructionKind::VirtualChangeDivisorW
        } else {
            InstructionKind::VirtualChangeDivisor
        },
        t0,
        dividend,
        divisor,
    )?;

    if word {
        asm.emit_i(InstructionKind::VirtualSignExtendWord, t1, a2, 0)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t1, a2, 0)?;
        asm.emit_i(InstructionKind::SRAI, t2, a3, 32)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t2, 0, 0)?;
    } else {
        asm.emit_r(InstructionKind::MULH, t1, a2, t0)?;
        t2 = asm.allocator().allocate()?;
        t3 = asm.allocator().allocate()?;
        asm.emit_r(InstructionKind::MUL, t2, a2, t0)?;
        asm.emit_i(InstructionKind::SRAI, t3, t2, shmat)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t1, t3, 0)?;
    }

    if word {
        asm.emit_i(InstructionKind::SRAI, t2, dividend, shmat)?;
        asm.emit_r(InstructionKind::XOR, t3, a3, t2)?;
        asm.emit_r(InstructionKind::SUB, t3, t3, t2)?;
        asm.emit_r(InstructionKind::MUL, t1, a2, t0)?;
        asm.emit_r(InstructionKind::ADD, t1, t1, t3)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t1, dividend, 0)?;
        asm.emit_i(InstructionKind::SRAI, t2, t0, 31)?;
        asm.emit_r(InstructionKind::XOR, t1, t0, t2)?;
        asm.emit_r(InstructionKind::SUB, t1, t1, t2)?;
        asm.emit_b(
            InstructionKind::VirtualAssertValidUnsignedRemainder,
            a3,
            t1,
            0,
        )?;
        asm.emit_i(
            InstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            if remainder_output { t3 } else { a2 },
            0,
        )?;
    } else {
        asm.emit_i(InstructionKind::SRAI, t1, dividend, shmat)?;
        asm.emit_r(InstructionKind::XOR, t3, a3, t1)?;
        asm.emit_r(InstructionKind::SUB, t3, t3, t1)?;
        asm.emit_r(InstructionKind::ADD, t2, t2, t3)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t2, a0, 0)?;
        asm.emit_i(InstructionKind::SRAI, t1, t0, shmat)?;
        asm.emit_r(
            InstructionKind::XOR,
            if remainder_output { t2 } else { t3 },
            t0,
            t1,
        )?;
        let abs_divisor = if remainder_output { t2 } else { t3 };
        asm.emit_r(InstructionKind::SUB, abs_divisor, abs_divisor, t1)?;
        asm.emit_b(
            InstructionKind::VirtualAssertValidUnsignedRemainder,
            a3,
            abs_divisor,
            0,
        )?;
        asm.emit_i(
            InstructionKind::ADDI,
            rd(instruction)?,
            if remainder_output { t3 } else { a2 },
            0,
        )?;
    }

    let sequence = asm.finalize()?;
    allocator.release(a2)?;
    allocator.release(a3)?;
    allocator.release(t0)?;
    allocator.release(t1)?;
    allocator.release(t2)?;
    allocator.release(t3)?;
    if let Some(t4) = t4 {
        allocator.release(t4)?;
    }
    Ok(sequence)
}

fn expand_divu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_j(InstructionKind::VirtualAdvice, v0, 0)?;
    asm.emit_b(
        InstructionKind::VirtualAssertValidDiv0,
        rs2(instruction)?,
        v0,
        0,
    )?;
    asm.emit_b(
        InstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(InstructionKind::MUL, v1, v0, rs2(instruction)?)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v1, rs1(instruction)?, 0)?;
    asm.emit_r(InstructionKind::SUB, v1, rs1(instruction)?, v1)?;
    asm.emit_b(
        InstructionKind::VirtualAssertValidUnsignedRemainder,
        v1,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

fn expand_remu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_j(InstructionKind::VirtualAdvice, v0, 0)?;
    asm.emit_b(
        InstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(InstructionKind::MUL, v0, v0, rs2(instruction)?)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v0, rs1(instruction)?, 0)?;
    asm.emit_r(InstructionKind::SUB, v0, rs1(instruction)?, v0)?;
    asm.emit_b(
        InstructionKind::VirtualAssertValidUnsignedRemainder,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    Ok(sequence)
}

fn expand_divw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_signed_div_rem(instruction, allocator, true, false)
}

fn expand_remw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_signed_div_rem(instruction, allocator, true, true)
}

fn expand_divuw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_unsigned_word_div_rem(instruction, allocator, false)
}

fn expand_remuw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_unsigned_word_div_rem(instruction, allocator, true)
}

fn expand_unsigned_word_div_rem(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    remainder_output: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let rs1_extended = allocator.allocate()?;
    let rs2_extended = allocator.allocate()?;
    let quotient = allocator.allocate()?;
    let tmp = if remainder_output {
        quotient
    } else {
        allocator.allocate()?
    };
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualZeroExtendWord,
        rs1_extended,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(
        InstructionKind::VirtualZeroExtendWord,
        rs2_extended,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_j(InstructionKind::VirtualAdvice, quotient, 0)?;
    asm.emit_b(
        InstructionKind::VirtualAssertMulUNoOverflow,
        quotient,
        rs2_extended,
        0,
    )?;
    asm.emit_r(InstructionKind::MUL, tmp, quotient, rs2_extended)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, tmp, rs1_extended, 0)?;
    asm.emit_r(InstructionKind::SUB, tmp, rs1_extended, tmp)?;
    asm.emit_b(
        InstructionKind::VirtualAssertValidUnsignedRemainder,
        tmp,
        rs2_extended,
        0,
    )?;
    if remainder_output {
        asm.emit_i(
            InstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            tmp,
            0,
        )?;
    } else {
        asm.emit_i(InstructionKind::VirtualSignExtendWord, tmp, quotient, 0)?;
        asm.emit_b(
            InstructionKind::VirtualAssertValidDiv0,
            rs2_extended,
            tmp,
            0,
        )?;
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, tmp, 0)?;
    }
    let sequence = asm.finalize()?;
    allocator.release(rs1_extended)?;
    allocator.release(rs2_extended)?;
    allocator.release(quotient)?;
    if !remainder_output {
        allocator.release(tmp)?;
    }
    Ok(sequence)
}

fn expand_sb(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_narrow_store(instruction, allocator, 0xff, None)
}

fn expand_sh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_narrow_store(
        instruction,
        allocator,
        0xffff,
        Some(InstructionKind::VirtualAssertHalfwordAlignment),
    )
}

fn expand_narrow_store(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    mask: i128,
    alignment: Option<InstructionKind>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    if let Some(alignment) = alignment {
        asm.emit_align(alignment, rs1(instruction)?, instruction.operands.imm)?;
    }
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v2, v1, 0)?;
    asm.emit_i(InstructionKind::SLLI, v3, v0, 3)?;
    asm.emit_u(InstructionKind::LUI, v0, mask)?;
    asm.emit_r(InstructionKind::SLL, v0, v0, v3)?;
    asm.emit_r(InstructionKind::SLL, v3, rs2(instruction)?, v3)?;
    asm.emit_r(InstructionKind::XOR, v3, v2, v3)?;
    asm.emit_r(InstructionKind::AND, v3, v3, v0)?;
    asm.emit_r(InstructionKind::XOR, v2, v2, v3)?;
    asm.emit_s(InstructionKind::SD, v1, v2, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    allocator.release(v3)?;
    Ok(sequence)
}

fn expand_sw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v2, v1, 0)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_i(InstructionKind::ORI, v3, 0, format_i_imm(-1))?;
    asm.emit_i(InstructionKind::SRLI, v3, v3, 32)?;
    asm.emit_r(InstructionKind::SLL, v3, v3, v0)?;
    asm.emit_r(InstructionKind::SLL, v0, rs2(instruction)?, v0)?;
    asm.emit_r(InstructionKind::XOR, v0, v2, v0)?;
    asm.emit_r(InstructionKind::AND, v0, v0, v3)?;
    asm.emit_r(InstructionKind::XOR, v2, v2, v0)?;
    asm.emit_s(InstructionKind::SD, v1, v2, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    allocator.release(v3)?;
    Ok(sequence)
}

fn expand_scd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    let v_success = asm.allocator().allocate()?;
    asm.emit_j(InstructionKind::VirtualAdvice, v_success, 0)?;

    let v_one = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::ADDI, v_one, 0, 1)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v_success, v_one, 0)?;
    asm.allocator().release(v_one)?;

    let v_addr_diff = asm.allocator().allocate()?;
    asm.emit_r(
        InstructionKind::SUB,
        v_addr_diff,
        v_reservation,
        rs1(instruction)?,
    )?;
    asm.emit_r(InstructionKind::MUL, v_addr_diff, v_success, v_addr_diff)?;
    asm.emit_b(InstructionKind::VirtualAssertEQ, v_addr_diff, 0, 0)?;
    asm.allocator().release(v_addr_diff)?;

    let v_mem = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::LD, v_mem, rs1(instruction)?, 0)?;

    let v_diff = asm.allocator().allocate()?;
    asm.emit_r(InstructionKind::SUB, v_diff, rs2(instruction)?, v_mem)?;
    asm.emit_r(InstructionKind::MUL, v_diff, v_diff, v_success)?;
    asm.emit_r(InstructionKind::ADD, v_diff, v_mem, v_diff)?;
    asm.allocator().release(v_mem)?;

    asm.emit_s(InstructionKind::SD, rs1(instruction)?, v_diff, 0)?;
    asm.allocator().release(v_diff)?;

    asm.emit_i(InstructionKind::ADDI, v_reservation, 0, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, 0, 0)?;
    asm.emit_i(InstructionKind::XORI, rd(instruction)?, v_success, 1)?;
    asm.allocator().release(v_success)?;

    asm.finalize()
}

fn expand_scw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_w_register();
    let v_reservation_d = allocator.reservation_d_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    let v_success = asm.allocator().allocate()?;
    asm.emit_j(InstructionKind::VirtualAdvice, v_success, 0)?;

    let v_one = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::ADDI, v_one, 0, 1)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v_success, v_one, 0)?;
    asm.allocator().release(v_one)?;

    let v_addr_diff = asm.allocator().allocate()?;
    asm.emit_r(
        InstructionKind::SUB,
        v_addr_diff,
        v_reservation,
        rs1(instruction)?,
    )?;
    asm.emit_r(InstructionKind::MUL, v_addr_diff, v_success, v_addr_diff)?;
    asm.emit_b(InstructionKind::VirtualAssertEQ, v_addr_diff, 0, 0)?;
    asm.allocator().release(v_addr_diff)?;

    asm.emit_i(InstructionKind::ADDI, v_reservation, v_success, 0)?;
    asm.allocator().release(v_success)?;

    let v_mem = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::LW, v_mem, rs1(instruction)?, 0)?;

    let v_diff = asm.allocator().allocate()?;
    asm.emit_r(InstructionKind::SUB, v_diff, rs2(instruction)?, v_mem)?;
    asm.emit_r(InstructionKind::MUL, v_diff, v_diff, v_reservation)?;
    asm.emit_r(InstructionKind::ADD, v_diff, v_mem, v_diff)?;
    asm.allocator().release(v_mem)?;

    asm.emit_i(InstructionKind::ADDI, v_reservation_d, v_diff, 0)?;
    asm.allocator().release(v_diff)?;

    asm.emit_s(InstructionKind::SW, rs1(instruction)?, v_reservation_d, 0)?;

    asm.emit_i(InstructionKind::XORI, rd(instruction)?, v_reservation, 1)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation, 0, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, 0, 0)?;

    asm.finalize()
}

fn expand_slli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        rd(instruction)?,
        rs1(instruction)?,
        1i128 << shift,
    )?;
    asm.finalize()
}

fn expand_csrrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    if csr == 0 {
        return Ok(vec![NormalizedInstruction::default()]);
    }
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

fn expand_csrrs(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    if csr == 0 {
        return Ok(vec![NormalizedInstruction::default()]);
    }
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

fn expand_ebreak(
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

fn expand_ecall(
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

fn expand_mret(
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

fn expand_sll(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualPow2, v_pow2, rs2(instruction)?, 0)?;
    asm.emit_r(
        InstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_pow2)?;
    Ok(sequence)
}

fn expand_slliw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x1f;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        rd(instruction)?,
        rs1(instruction)?,
        1i128 << shift,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    asm.finalize()
}

fn expand_sllw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualPow2W, v_pow2, rs2(instruction)?, 0)?;
    asm.emit_r(
        InstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_pow2)?;
    Ok(sequence)
}

fn expand_srl(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRL,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}

fn expand_srlw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let v_rs1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    )?;
    asm.emit_i(InstructionKind::ORI, v_bitmask, rs2(instruction)?, 32)?;
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        v_bitmask,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRL,
        rd(instruction)?,
        v_rs1,
        v_bitmask,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_bitmask)?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}

fn expand_sra(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRA,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}

fn expand_sraw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        v_rs1,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(InstructionKind::ANDI, v_bitmask, rs2(instruction)?, 0x1f)?;
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        v_bitmask,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRA,
        rd(instruction)?,
        v_rs1,
        v_bitmask,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}

fn expand_srli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSRLI,
        rd(instruction)?,
        rs1(instruction)?,
        bitmask as i128,
    )?;
    asm.finalize()
}

fn expand_srliw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = (instruction.operands.imm & 0x1f) + 32;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSRLI,
        rd(instruction)?,
        v_rs1,
        bitmask as i128,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}

fn expand_srai(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSRAI,
        rd(instruction)?,
        rs1(instruction)?,
        bitmask as i128,
    )?;
    asm.finalize()
}

fn expand_sraiw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = instruction.operands.imm & 0x1f;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        v_rs1,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSRAI,
        rd(instruction)?,
        v_rs1,
        bitmask as i128,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}

fn right_shift_bitmask(shift: u32, len: u32) -> u64 {
    let ones = (1u128 << (len - shift)) - 1;
    (ones << shift) as u64
}

fn rd(instruction: &NormalizedInstruction) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rd
        .ok_or(ExpansionError::MalformedInstruction("missing rd"))
}

fn rs1(instruction: &NormalizedInstruction) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rs1
        .ok_or(ExpansionError::MalformedInstruction("missing rs1"))
}

fn rs2(instruction: &NormalizedInstruction) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rs2
        .ok_or(ExpansionError::MalformedInstruction("missing rs2"))
}

fn format_i_imm(imm: i128) -> i128 {
    (imm as i64 as u64) as i128
}

fn csr_address(instruction: &NormalizedInstruction) -> u16 {
    (instruction.operands.imm & 0xfff) as u16
}

const fn handles_rd_zero_internally(instruction_kind: InstructionKind) -> bool {
    matches!(
        instruction_kind,
        InstructionKind::ECALL
            | InstructionKind::MRET
            | InstructionKind::EBREAK
            | InstructionKind::CSRRW
            | InstructionKind::CSRRS
    )
}

const fn has_side_effects(instruction_kind: InstructionKind) -> bool {
    matches!(
        instruction_kind,
        InstructionKind::AdviceLB
            | InstructionKind::AdviceLD
            | InstructionKind::AdviceLH
            | InstructionKind::AdviceLW
            | InstructionKind::AMOADDD
            | InstructionKind::AMOADDW
            | InstructionKind::AMOANDD
            | InstructionKind::AMOANDW
            | InstructionKind::AMOMAXD
            | InstructionKind::AMOMAXUD
            | InstructionKind::AMOMAXUW
            | InstructionKind::AMOMAXW
            | InstructionKind::AMOMIND
            | InstructionKind::AMOMINUD
            | InstructionKind::AMOMINUW
            | InstructionKind::AMOMINW
            | InstructionKind::AMOORD
            | InstructionKind::AMOORW
            | InstructionKind::AMOSWAPD
            | InstructionKind::AMOSWAPW
            | InstructionKind::AMOXORD
            | InstructionKind::AMOXORW
            | InstructionKind::BEQ
            | InstructionKind::BGE
            | InstructionKind::BGEU
            | InstructionKind::BLT
            | InstructionKind::BLTU
            | InstructionKind::BNE
            | InstructionKind::CSRRS
            | InstructionKind::CSRRW
            | InstructionKind::EBREAK
            | InstructionKind::ECALL
            | InstructionKind::Inline
            | InstructionKind::JAL
            | InstructionKind::JALR
            | InstructionKind::LB
            | InstructionKind::LBU
            | InstructionKind::LD
            | InstructionKind::LH
            | InstructionKind::LHU
            | InstructionKind::LRD
            | InstructionKind::LRW
            | InstructionKind::LW
            | InstructionKind::LWU
            | InstructionKind::MRET
            | InstructionKind::SB
            | InstructionKind::SCD
            | InstructionKind::SCW
            | InstructionKind::SD
            | InstructionKind::SH
            | InstructionKind::SW
            | InstructionKind::VirtualAdviceLoad
            | InstructionKind::VirtualHostIO
            | InstructionKind::VirtualSW
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instruction(
        instruction_kind: InstructionKind,
        rd: Option<u8>,
        is_compressed: bool,
    ) -> NormalizedInstruction {
        NormalizedInstruction {
            instruction_kind,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd,
                rs1: Some(1),
                rs2: Some(2),
                imm: 7,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed,
        }
    }

    #[test]
    fn side_effect_free_rd_zero_becomes_noop_addi() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let expanded = expand_instruction(
            &instruction(InstructionKind::ADD, Some(0), true),
            &mut allocator,
        )?;

        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].instruction_kind, InstructionKind::ADDI);
        assert_eq!(expanded[0].operands.rd, Some(0));
        assert_eq!(expanded[0].operands.rs1, Some(0));
        assert_eq!(expanded[0].operands.rs2, None);
        assert_eq!(expanded[0].operands.imm, 0);
        assert!(expanded[0].is_compressed);
        Ok(())
    }

    #[test]
    fn side_effecting_rd_zero_rewrites_to_temporary_register() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let expanded = expand_instruction(
            &instruction(InstructionKind::JAL, Some(0), false),
            &mut allocator,
        )?;

        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].instruction_kind, InstructionKind::JAL);
        assert_eq!(expanded[0].operands.rd, Some(40));
        Ok(())
    }

    #[test]
    fn trap_related_rd_zero_uses_instruction_expansion() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let input = instruction(InstructionKind::ECALL, Some(0), false);
        let expanded = expand_instruction(&input, &mut allocator)?;

        assert_eq!(expanded.len(), 7);
        assert_eq!(expanded[0].instruction_kind, InstructionKind::AUIPC);
        assert_eq!(expanded[6].instruction_kind, InstructionKind::JALR);
        Ok(())
    }

    #[test]
    fn inline_requires_provider() {
        let mut allocator = ExpansionAllocator::new();
        let input = instruction(InstructionKind::Inline, Some(3), false);

        assert!(matches!(
            expand_instruction(&input, &mut allocator),
            Err(ExpansionError::InlineProviderRequired)
        ));
    }

    #[test]
    fn inline_rd_zero_is_remapped_before_provider() -> Result<(), ExpansionError> {
        #[derive(Default)]
        struct CapturingProvider {
            captured: Option<NormalizedInstruction>,
        }

        impl InlineExpansionProvider for CapturingProvider {
            fn expand_inline(
                &mut self,
                instruction: &NormalizedInstruction,
                _allocator: &mut ExpansionAllocator,
            ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
                self.captured = Some(*instruction);
                Ok(vec![*instruction])
            }
        }

        let input = NormalizedInstruction {
            instruction_kind: InstructionKind::Inline,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(0),
                rs1: Some(10),
                rs2: Some(20),
                imm: 0x0b,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        let mut allocator = ExpansionAllocator::new();
        let mut provider = CapturingProvider::default();

        let expanded = expand_instruction_with_provider(&input, &mut allocator, &mut provider)?;

        let mut expected = input;
        expected.operands.rd = Some(40);

        assert_eq!(provider.captured, Some(expected));
        assert_eq!(expanded, vec![expected]);
        Ok(())
    }
}
