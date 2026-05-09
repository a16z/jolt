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

use allocator::{
    csr_to_virtual_register, mcause_register, mepc_register, mstatus_register, mtval_register,
    reservation_d_register, reservation_w_register, trap_handler_register,
};
use arithmetic::*;
use control_flow::*;
use core::ExpansionState;
use division::*;
use grammar::{
    is_source_only, reg, ExpandedInstructionSequence, ExpansionBuilder, RegisterOperand, TempId,
};
use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};
use memory::*;
use metadata::stamp_sequence;
use operands::*;
use shifts::*;

pub trait InlineExpansionProvider {
    /// Expands a registered inline row into normalized rows.
    ///
    /// Provider output intentionally stays outside the provider-free builder
    /// core. The top-level entry point remaps `rd = x0` before calling this
    /// hook, then validates target legality and stamps sequence metadata.
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
        let rows = inline_provider.expand_inline(instruction, allocator)?;
        return finalize_inline_provider_rows(*instruction, allocator, rows);
    }

    let owned_allocator = std::mem::take(allocator);
    let mut state = ExpansionState::new(owned_allocator);
    let result = state.expand_one_core(instruction);
    *allocator = state.into_allocator();
    result
}

fn finalize_inline_provider_rows(
    source: NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    mut rows: Vec<NormalizedInstruction>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    for register in allocator.take_registers_for_reset()? {
        rows.push(NormalizedInstruction {
            instruction_kind: JoltInstructionKind::ADDI,
            address: source.address,
            operands: NormalizedOperands {
                rd: Some(register),
                rs1: Some(0),
                rs2: None,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        });
    }
    stamp_sequence(rows, source.is_compressed)
}

fn expand_instruction_core(
    instruction: &NormalizedInstruction,
    state: &mut ExpansionState,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    if instruction.operands.rd == Some(0)
        && !handles_rd_zero_internally(instruction.instruction_kind)
    {
        if instruction.instruction_kind.has_side_effects() {
            let virtual_register = state.allocate_register()?;
            let mut rewritten = *instruction;
            rewritten.operands.rd = Some(virtual_register);
            let expanded = state.expand_one_core(&rewritten);
            state.release_register(virtual_register)?;
            return expanded;
        }
        return Ok(vec![noop_for(*instruction)]);
    }

    if instruction.instruction_kind == JoltInstructionKind::Inline {
        return Err(ExpansionError::InlineProviderRequired);
    }
    if !is_source_only(instruction.instruction_kind) {
        return Ok(vec![*instruction]);
    }
    let sequence = expand_source_only_instruction(instruction)?;
    state.materialize(sequence)
}

fn expand_source_only_instruction(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    match instruction.instruction_kind {
        JoltInstructionKind::ADDIW => expand_addiw(instruction),
        JoltInstructionKind::ADDW => expand_addw(instruction),
        JoltInstructionKind::SUBW => expand_subw(instruction),
        JoltInstructionKind::MULH => expand_mulh(instruction),
        JoltInstructionKind::MULHSU => expand_mulhsu(instruction),
        JoltInstructionKind::MULW => expand_mulw(instruction),
        JoltInstructionKind::LB => expand_lb(instruction),
        JoltInstructionKind::LBU => expand_lbu(instruction),
        JoltInstructionKind::LH => expand_lh(instruction),
        JoltInstructionKind::LHU => expand_lhu(instruction),
        JoltInstructionKind::LW => expand_lw(instruction),
        JoltInstructionKind::LWU => expand_lwu(instruction),
        JoltInstructionKind::AdviceLB => expand_advice_lb(instruction),
        JoltInstructionKind::AdviceLH => expand_advice_lh(instruction),
        JoltInstructionKind::AdviceLW => expand_advice_lw(instruction),
        JoltInstructionKind::AdviceLD => expand_advice_ld(instruction),
        JoltInstructionKind::AMOADDD => expand_amoaddd(instruction),
        JoltInstructionKind::AMOANDD => expand_amoandd(instruction),
        JoltInstructionKind::AMOORD => expand_amoord(instruction),
        JoltInstructionKind::AMOXORD => expand_amoxord(instruction),
        JoltInstructionKind::AMOSWAPD => expand_amoswapd(instruction),
        JoltInstructionKind::AMOMAXD => expand_amomaxd(instruction),
        JoltInstructionKind::AMOMAXUD => expand_amomaxud(instruction),
        JoltInstructionKind::AMOMIND => expand_amomind(instruction),
        JoltInstructionKind::AMOMINUD => expand_amominud(instruction),
        JoltInstructionKind::AMOADDW => expand_amoaddw(instruction),
        JoltInstructionKind::AMOANDW => expand_amoandw(instruction),
        JoltInstructionKind::AMOORW => expand_amoorw(instruction),
        JoltInstructionKind::AMOXORW => expand_amoxorw(instruction),
        JoltInstructionKind::AMOSWAPW => expand_amoswapw(instruction),
        JoltInstructionKind::AMOMAXW => expand_amomaxw(instruction),
        JoltInstructionKind::AMOMAXUW => expand_amomaxuw(instruction),
        JoltInstructionKind::AMOMINW => expand_amominw(instruction),
        JoltInstructionKind::AMOMINUW => expand_amominuw(instruction),
        JoltInstructionKind::LRD => expand_lrd(instruction),
        JoltInstructionKind::LRW => expand_lrw(instruction),
        JoltInstructionKind::DIV => expand_div(instruction),
        JoltInstructionKind::DIVU => expand_divu(instruction),
        JoltInstructionKind::DIVW => expand_divw(instruction),
        JoltInstructionKind::DIVUW => expand_divuw(instruction),
        JoltInstructionKind::REM => expand_rem(instruction),
        JoltInstructionKind::REMU => expand_remu(instruction),
        JoltInstructionKind::REMW => expand_remw(instruction),
        JoltInstructionKind::REMUW => expand_remuw(instruction),
        JoltInstructionKind::SB => expand_sb(instruction),
        JoltInstructionKind::SCD => expand_scd(instruction),
        JoltInstructionKind::SCW => expand_scw(instruction),
        JoltInstructionKind::SH => expand_sh(instruction),
        JoltInstructionKind::SW => expand_sw(instruction),
        JoltInstructionKind::CSRRW => expand_csrrw(instruction),
        JoltInstructionKind::CSRRS => expand_csrrs(instruction),
        JoltInstructionKind::EBREAK => expand_ebreak(instruction),
        JoltInstructionKind::ECALL => expand_ecall(instruction),
        JoltInstructionKind::MRET => expand_mret(instruction),
        JoltInstructionKind::SLL => expand_sll(instruction),
        JoltInstructionKind::SLLI => expand_slli(instruction),
        JoltInstructionKind::SLLW => expand_sllw(instruction),
        JoltInstructionKind::SLLIW => expand_slliw(instruction),
        JoltInstructionKind::SRL => expand_srl(instruction),
        JoltInstructionKind::SRLI => expand_srli(instruction),
        JoltInstructionKind::SRA => expand_sra(instruction),
        JoltInstructionKind::SRAI => expand_srai(instruction),
        JoltInstructionKind::SRLIW => expand_srliw(instruction),
        JoltInstructionKind::SRAIW => expand_sraiw(instruction),
        JoltInstructionKind::SRLW => expand_srlw(instruction),
        JoltInstructionKind::SRAW => expand_sraw(instruction),
        _ => Err(ExpansionError::UnsupportedInstruction),
    }
}

pub fn expand_program(
    instructions: &[NormalizedInstruction],
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_program_with_provider(instructions, &mut NoInlineExpansionProvider)
}

pub fn expand_program_with_provider<P: InlineExpansionProvider + ?Sized>(
    instructions: &[NormalizedInstruction],
    inline_provider: &mut P,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let mut expanded = Vec::new();
    for instruction in instructions {
        expanded.extend(expand_instruction_with_provider(
            instruction,
            &mut allocator,
            inline_provider,
        )?);
    }
    Ok(expanded)
}
