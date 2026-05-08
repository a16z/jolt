//! RV64 bytecode expansion from decoded source rows into final Jolt bytecode.
//!
//! Expansion intentionally has no `Xlen` parameter: the `jolt-program` pipeline
//! only supports RV64. RV32/ELF32 inputs are rejected before this module is
//! called.

pub mod allocator;
mod arithmetic;
mod control_flow;
mod core;
mod division;
pub mod error;
mod memory;
mod operands;
mod shifts;
#[cfg(test)]
mod tests;

pub use allocator::ExpansionAllocator;
pub use error::ExpansionError;

use arithmetic::*;
use control_flow::*;
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

    match instruction.instruction_kind {
        JoltInstructionKind::Inline => inline_provider.expand_inline(instruction, allocator),
        JoltInstructionKind::ADDIW => expand_addiw(instruction, allocator),
        JoltInstructionKind::ADDW => expand_addw(instruction, allocator),
        JoltInstructionKind::SUBW => expand_subw(instruction, allocator),
        JoltInstructionKind::MULH => expand_mulh(instruction, allocator),
        JoltInstructionKind::MULHSU => expand_mulhsu(instruction, allocator),
        JoltInstructionKind::MULW => expand_mulw(instruction, allocator),
        JoltInstructionKind::LB => expand_lb(instruction, allocator),
        JoltInstructionKind::LBU => expand_lbu(instruction, allocator),
        JoltInstructionKind::LH => expand_lh(instruction, allocator),
        JoltInstructionKind::LHU => expand_lhu(instruction, allocator),
        JoltInstructionKind::LW => expand_lw(instruction, allocator),
        JoltInstructionKind::LWU => expand_lwu(instruction, allocator),
        JoltInstructionKind::AdviceLB => expand_advice_lb(instruction, allocator),
        JoltInstructionKind::AdviceLH => expand_advice_lh(instruction, allocator),
        JoltInstructionKind::AdviceLW => expand_advice_lw(instruction, allocator),
        JoltInstructionKind::AdviceLD => expand_advice_ld(instruction, allocator),
        JoltInstructionKind::AMOADDD => expand_amoaddd(instruction, allocator),
        JoltInstructionKind::AMOANDD => expand_amoandd(instruction, allocator),
        JoltInstructionKind::AMOORD => expand_amoord(instruction, allocator),
        JoltInstructionKind::AMOXORD => expand_amoxord(instruction, allocator),
        JoltInstructionKind::AMOSWAPD => expand_amoswapd(instruction, allocator),
        JoltInstructionKind::AMOMAXD => expand_amomaxd(instruction, allocator),
        JoltInstructionKind::AMOMAXUD => expand_amomaxud(instruction, allocator),
        JoltInstructionKind::AMOMIND => expand_amomind(instruction, allocator),
        JoltInstructionKind::AMOMINUD => expand_amominud(instruction, allocator),
        JoltInstructionKind::AMOADDW => expand_amoaddw(instruction, allocator),
        JoltInstructionKind::AMOANDW => expand_amoandw(instruction, allocator),
        JoltInstructionKind::AMOORW => expand_amoorw(instruction, allocator),
        JoltInstructionKind::AMOXORW => expand_amoxorw(instruction, allocator),
        JoltInstructionKind::AMOSWAPW => expand_amoswapw(instruction, allocator),
        JoltInstructionKind::AMOMAXW => expand_amomaxw(instruction, allocator),
        JoltInstructionKind::AMOMAXUW => expand_amomaxuw(instruction, allocator),
        JoltInstructionKind::AMOMINW => expand_amominw(instruction, allocator),
        JoltInstructionKind::AMOMINUW => expand_amominuw(instruction, allocator),
        JoltInstructionKind::LRD => expand_lrd(instruction, allocator),
        JoltInstructionKind::LRW => expand_lrw(instruction, allocator),
        JoltInstructionKind::DIV => expand_div(instruction, allocator),
        JoltInstructionKind::DIVU => expand_divu(instruction, allocator),
        JoltInstructionKind::DIVW => expand_divw(instruction, allocator),
        JoltInstructionKind::DIVUW => expand_divuw(instruction, allocator),
        JoltInstructionKind::REM => expand_rem(instruction, allocator),
        JoltInstructionKind::REMU => expand_remu(instruction, allocator),
        JoltInstructionKind::REMW => expand_remw(instruction, allocator),
        JoltInstructionKind::REMUW => expand_remuw(instruction, allocator),
        JoltInstructionKind::SB => expand_sb(instruction, allocator),
        JoltInstructionKind::SCD => expand_scd(instruction, allocator),
        JoltInstructionKind::SCW => expand_scw(instruction, allocator),
        JoltInstructionKind::SH => expand_sh(instruction, allocator),
        JoltInstructionKind::SW => expand_sw(instruction, allocator),
        JoltInstructionKind::CSRRW => expand_csrrw(instruction, allocator),
        JoltInstructionKind::CSRRS => expand_csrrs(instruction, allocator),
        JoltInstructionKind::EBREAK => expand_ebreak(instruction, allocator),
        JoltInstructionKind::ECALL => expand_ecall(instruction, allocator),
        JoltInstructionKind::MRET => expand_mret(instruction, allocator),
        JoltInstructionKind::SLL => expand_sll(instruction, allocator),
        JoltInstructionKind::SLLI => expand_slli(instruction, allocator),
        JoltInstructionKind::SLLW => expand_sllw(instruction, allocator),
        JoltInstructionKind::SLLIW => expand_slliw(instruction, allocator),
        JoltInstructionKind::SRL => expand_srl(instruction, allocator),
        JoltInstructionKind::SRLI => expand_srli(instruction, allocator),
        JoltInstructionKind::SRA => expand_sra(instruction, allocator),
        JoltInstructionKind::SRAI => expand_srai(instruction, allocator),
        JoltInstructionKind::SRLIW => expand_srliw(instruction, allocator),
        JoltInstructionKind::SRAIW => expand_sraiw(instruction, allocator),
        JoltInstructionKind::SRLW => expand_srlw(instruction, allocator),
        JoltInstructionKind::SRAW => expand_sraw(instruction, allocator),
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
