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
mod operands;
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
        InstructionKind::AdviceLB => expand_advice_lb(instruction, allocator),
        InstructionKind::AdviceLH => expand_advice_lh(instruction, allocator),
        InstructionKind::AdviceLW => expand_advice_lw(instruction, allocator),
        InstructionKind::AdviceLD => expand_advice_ld(instruction, allocator),
        InstructionKind::AMOADDD => expand_amoaddd(instruction, allocator),
        InstructionKind::AMOANDD => expand_amoandd(instruction, allocator),
        InstructionKind::AMOORD => expand_amoord(instruction, allocator),
        InstructionKind::AMOXORD => expand_amoxord(instruction, allocator),
        InstructionKind::AMOSWAPD => expand_amoswapd(instruction, allocator),
        InstructionKind::AMOMAXD => expand_amomaxd(instruction, allocator),
        InstructionKind::AMOMAXUD => expand_amomaxud(instruction, allocator),
        InstructionKind::AMOMIND => expand_amomind(instruction, allocator),
        InstructionKind::AMOMINUD => expand_amominud(instruction, allocator),
        InstructionKind::AMOADDW => expand_amoaddw(instruction, allocator),
        InstructionKind::AMOANDW => expand_amoandw(instruction, allocator),
        InstructionKind::AMOORW => expand_amoorw(instruction, allocator),
        InstructionKind::AMOXORW => expand_amoxorw(instruction, allocator),
        InstructionKind::AMOSWAPW => expand_amoswapw(instruction, allocator),
        InstructionKind::AMOMAXW => expand_amomaxw(instruction, allocator),
        InstructionKind::AMOMAXUW => expand_amomaxuw(instruction, allocator),
        InstructionKind::AMOMINW => expand_amominw(instruction, allocator),
        InstructionKind::AMOMINUW => expand_amominuw(instruction, allocator),
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
