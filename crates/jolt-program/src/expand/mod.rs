//! RV64 bytecode expansion from decoded source rows into final Jolt bytecode.
//!
//! Expansion intentionally has no `Xlen` parameter: the `jolt-program` pipeline
//! only supports RV64. RV32/ELF32 inputs should be rejected before this module is
//! called.

pub mod allocator;
mod arithmetic;
pub mod assembler;
mod control_flow;
mod division;
pub mod error;
mod memory;
pub mod metadata;
mod operands;
pub mod sequences;
mod shifts;
#[cfg(test)]
mod tests;

pub use allocator::ExpansionAllocator;
pub use error::ExpansionError;

use arithmetic::*;
use control_flow::*;
use division::*;
use jolt_riscv::{InstructionKind, NormalizedInstruction, NormalizedOperands};
use memory::*;
use operands::*;
use shifts::*;

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
