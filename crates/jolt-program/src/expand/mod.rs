//! RV64 bytecode expansion from decoded source rows into final Jolt bytecode.
//!
//! Expansion intentionally has no `Xlen` parameter: the `jolt-program` pipeline
//! only supports RV64. RV32/ELF32 inputs are rejected before this module is
//! called.

pub mod allocator;
mod arithmetic;
mod buffer;
mod control_flow;
mod core;
mod division;
pub mod error;
mod grammar;
mod memory;
mod metadata;
mod operands;
mod shifts;
#[cfg(test)]
mod tests;

pub use allocator::ExpansionAllocator;
pub use error::ExpansionError;

use arithmetic::*;
use control_flow::*;
use core::ExpansionState;
use division::*;
use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};
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
        if instruction.instruction_kind.has_side_effects() {
            let virtual_register = allocator.allocate()?;
            let mut rewritten = *instruction;
            rewritten.operands.rd = Some(virtual_register);
            let expanded = expand_instruction_with_provider(&rewritten, allocator, inline_provider);
            allocator.release(virtual_register)?;
            return expanded;
        }
        return Ok(vec![noop_for(*instruction)]);
    }

    if instruction.instruction_kind == JoltInstructionKind::Inline {
        return inline_provider.expand_inline(instruction, allocator);
    }

    ExpansionState::new(allocator).expand_one_core(instruction)
}

fn expand_instruction_core(
    instruction: &NormalizedInstruction,
    state: &mut ExpansionState<'_>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    if instruction.operands.rd == Some(0)
        && !handles_rd_zero_internally(instruction.instruction_kind)
    {
        if instruction.instruction_kind.has_side_effects() {
            let virtual_register = state.allocator().allocate()?;
            let mut rewritten = *instruction;
            rewritten.operands.rd = Some(virtual_register);
            let expanded = state.expand_one_core(&rewritten);
            state.allocator().release(virtual_register)?;
            return expanded;
        }
        return Ok(vec![noop_for(*instruction)]);
    }

    match instruction.instruction_kind {
        JoltInstructionKind::Inline => Err(ExpansionError::InlineProviderRequired),
        JoltInstructionKind::ADDIW => expand_addiw(instruction, state.allocator()),
        JoltInstructionKind::ADDW => expand_addw(instruction, state.allocator()),
        JoltInstructionKind::SUBW => expand_subw(instruction, state.allocator()),
        JoltInstructionKind::MULH => expand_mulh(instruction, state.allocator()),
        JoltInstructionKind::MULHSU => expand_mulhsu(instruction, state.allocator()),
        JoltInstructionKind::MULW => expand_mulw(instruction, state.allocator()),
        JoltInstructionKind::LB => expand_lb(instruction, state.allocator()),
        JoltInstructionKind::LBU => expand_lbu(instruction, state.allocator()),
        JoltInstructionKind::LH => expand_lh(instruction, state.allocator()),
        JoltInstructionKind::LHU => expand_lhu(instruction, state.allocator()),
        JoltInstructionKind::LW => expand_lw(instruction, state.allocator()),
        JoltInstructionKind::LWU => expand_lwu(instruction, state.allocator()),
        JoltInstructionKind::AdviceLB => expand_advice_lb(instruction, state.allocator()),
        JoltInstructionKind::AdviceLH => expand_advice_lh(instruction, state.allocator()),
        JoltInstructionKind::AdviceLW => expand_advice_lw(instruction, state.allocator()),
        JoltInstructionKind::AdviceLD => expand_advice_ld(instruction, state.allocator()),
        JoltInstructionKind::AMOADDD => expand_amoaddd(instruction, state.allocator()),
        JoltInstructionKind::AMOANDD => expand_amoandd(instruction, state.allocator()),
        JoltInstructionKind::AMOORD => expand_amoord(instruction, state.allocator()),
        JoltInstructionKind::AMOXORD => expand_amoxord(instruction, state.allocator()),
        JoltInstructionKind::AMOSWAPD => expand_amoswapd(instruction, state.allocator()),
        JoltInstructionKind::AMOMAXD => expand_amomaxd(instruction, state.allocator()),
        JoltInstructionKind::AMOMAXUD => expand_amomaxud(instruction, state.allocator()),
        JoltInstructionKind::AMOMIND => expand_amomind(instruction, state.allocator()),
        JoltInstructionKind::AMOMINUD => expand_amominud(instruction, state.allocator()),
        JoltInstructionKind::AMOADDW => expand_amoaddw(instruction, state.allocator()),
        JoltInstructionKind::AMOANDW => expand_amoandw(instruction, state.allocator()),
        JoltInstructionKind::AMOORW => expand_amoorw(instruction, state.allocator()),
        JoltInstructionKind::AMOXORW => expand_amoxorw(instruction, state.allocator()),
        JoltInstructionKind::AMOSWAPW => expand_amoswapw(instruction, state.allocator()),
        JoltInstructionKind::AMOMAXW => expand_amomaxw(instruction, state.allocator()),
        JoltInstructionKind::AMOMAXUW => expand_amomaxuw(instruction, state.allocator()),
        JoltInstructionKind::AMOMINW => expand_amominw(instruction, state.allocator()),
        JoltInstructionKind::AMOMINUW => expand_amominuw(instruction, state.allocator()),
        JoltInstructionKind::LRD => expand_lrd(instruction, state.allocator()),
        JoltInstructionKind::LRW => expand_lrw(instruction, state.allocator()),
        JoltInstructionKind::DIV => expand_div(instruction, state.allocator()),
        JoltInstructionKind::DIVU => expand_divu(instruction, state.allocator()),
        JoltInstructionKind::DIVW => expand_divw(instruction, state.allocator()),
        JoltInstructionKind::DIVUW => expand_divuw(instruction, state.allocator()),
        JoltInstructionKind::REM => expand_rem(instruction, state.allocator()),
        JoltInstructionKind::REMU => expand_remu(instruction, state.allocator()),
        JoltInstructionKind::REMW => expand_remw(instruction, state.allocator()),
        JoltInstructionKind::REMUW => expand_remuw(instruction, state.allocator()),
        JoltInstructionKind::SB => expand_sb(instruction, state.allocator()),
        JoltInstructionKind::SCD => expand_scd(instruction, state.allocator()),
        JoltInstructionKind::SCW => expand_scw(instruction, state.allocator()),
        JoltInstructionKind::SH => expand_sh(instruction, state.allocator()),
        JoltInstructionKind::SW => expand_sw(instruction, state.allocator()),
        JoltInstructionKind::CSRRW => expand_csrrw(instruction, state.allocator()),
        JoltInstructionKind::CSRRS => expand_csrrs(instruction, state.allocator()),
        JoltInstructionKind::EBREAK => expand_ebreak(instruction, state.allocator()),
        JoltInstructionKind::ECALL => expand_ecall(instruction, state.allocator()),
        JoltInstructionKind::MRET => expand_mret(instruction, state.allocator()),
        JoltInstructionKind::SLL => expand_sll(instruction, state.allocator()),
        JoltInstructionKind::SLLI => expand_slli(instruction, state.allocator()),
        JoltInstructionKind::SLLW => expand_sllw(instruction, state.allocator()),
        JoltInstructionKind::SLLIW => expand_slliw(instruction, state.allocator()),
        JoltInstructionKind::SRL => expand_srl(instruction, state.allocator()),
        JoltInstructionKind::SRLI => expand_srli(instruction, state.allocator()),
        JoltInstructionKind::SRA => expand_sra(instruction, state.allocator()),
        JoltInstructionKind::SRAI => expand_srai(instruction, state.allocator()),
        JoltInstructionKind::SRLIW => expand_srliw(instruction, state.allocator()),
        JoltInstructionKind::SRAIW => expand_sraiw(instruction, state.allocator()),
        JoltInstructionKind::SRLW => expand_srlw(instruction, state.allocator()),
        JoltInstructionKind::SRAW => expand_sraw(instruction, state.allocator()),
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
