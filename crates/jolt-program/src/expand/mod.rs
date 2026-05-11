//! RV64 bytecode expansion from decoded source rows into final Jolt bytecode.
//!
//! The pipeline has two phases:
//! 1. **Recipe building** (`grammar.rs`): each source-only instruction maps to a
//!    symbolic recipe — a sequence of `ExpansionOp`s referencing `TempId` placeholders.
//! 2. **Materialization** (`materialize.rs`): recipes are executed by binding temps to
//!    physical virtual registers, emitting concrete rows, and recursing for nested
//!    expansions.
//!
//! Expansion intentionally has no `Xlen` parameter: the `jolt-program` pipeline
//! only supports RV64. RV32/ELF32 inputs are rejected before this module is
//! called.

pub mod allocator;
mod arithmetic;
mod control_flow;
mod division;
pub mod error;
mod grammar;
mod materialize;
mod memory;
mod metadata;
mod operands;
mod shifts;
#[cfg(test)]
mod tests;

pub use allocator::ExpansionAllocator;
pub use error::ExpansionError;

use allocator::{
    mcause_register, mepc_register, mstatus_register, mtval_register, reservation_d_register,
    reservation_w_register, trap_handler_register, virtual_register_for_csr,
};
use arithmetic::*;
use control_flow::*;
use division::*;
use grammar::{reg, ExpandedInstructionSequence, ExpansionBuilder, RegisterOperand, TempId};
use jolt_riscv::{
    JoltInstructionKind, NormalizedInstruction, NormalizedOperands, SourceInstruction,
    SourceInstructionKind,
};
use materialize::ExpansionState;
use memory::*;
use metadata::stamp_inline_sequence;
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
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_instruction_with_provider(instruction, allocator, &mut NoInlineExpansionProvider)
}

pub fn expand_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut rewritten;
    let mut allocated_rd_zero_register = None;
    let instruction = if instruction.operands.rd == Some(0)
        && !instruction.instruction_kind.handles_rd_zero_internally()
    {
        if instruction.instruction_kind.has_side_effects() {
            let virtual_register = allocator.allocate()?;
            allocated_rd_zero_register = Some(virtual_register);
            rewritten = *instruction;
            rewritten.operands.rd = Some(virtual_register);
            &rewritten
        } else {
            return Ok(vec![noop_for_source(*instruction)]);
        }
    } else {
        instruction
    };

    let result = if instruction.instruction_kind == SourceInstructionKind::Inline {
        let normalized = instruction.into_normalized_instruction();
        let rows = inline_provider.expand_inline(&normalized, allocator)?;
        finalize_inline_provider_rows(normalized, allocator, rows)
    } else {
        let owned_allocator = std::mem::take(allocator);
        let mut state = ExpansionState::new(owned_allocator);
        let result = state.expand_source_recursive(instruction);
        *allocator = state.into_allocator();
        result
    };
    if let Some(register) = allocated_rd_zero_register {
        allocator.release(register)?;
    }
    result
}

#[cfg(test)]
fn expand_normalized_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut rewritten;
    let mut allocated_rd_zero_register = None;
    let instruction = if instruction.operands.rd == Some(0)
        && !handles_final_rd_zero_internally(instruction.instruction_kind)
    {
        if instruction.instruction_kind.has_side_effects() {
            let virtual_register = allocator.allocate()?;
            allocated_rd_zero_register = Some(virtual_register);
            rewritten = *instruction;
            rewritten.operands.rd = Some(virtual_register);
            &rewritten
        } else {
            return Ok(vec![noop_for(*instruction)]);
        }
    } else {
        instruction
    };

    let result = if instruction.instruction_kind == JoltInstructionKind::Inline {
        let rows = inline_provider.expand_inline(instruction, allocator)?;
        finalize_inline_provider_rows(*instruction, allocator, rows)
    } else {
        let owned_allocator = std::mem::take(allocator);
        let mut state = ExpansionState::new(owned_allocator);
        let result = state.expand_recursive(instruction);
        *allocator = state.into_allocator();
        result
    };
    if let Some(register) = allocated_rd_zero_register {
        allocator.release(register)?;
    }
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
    stamp_inline_sequence(rows, source.is_compressed)
}

/// Dispatches a source-only instruction to its recipe builder (phase 1).
fn expand_source_only_instruction(
    source: &SourceInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let instruction = source.into_normalized_instruction();
    match source.instruction_kind {
        SourceInstructionKind::ADDIW => expand_addiw(&instruction),
        SourceInstructionKind::ADDW => expand_addw(&instruction),
        SourceInstructionKind::SUBW => expand_subw(&instruction),
        SourceInstructionKind::MULH => expand_mulh(&instruction),
        SourceInstructionKind::MULHSU => expand_mulhsu(&instruction),
        SourceInstructionKind::MULW => expand_mulw(&instruction),
        SourceInstructionKind::LB => expand_lb(&instruction),
        SourceInstructionKind::LBU => expand_lbu(&instruction),
        SourceInstructionKind::LH => expand_lh(&instruction),
        SourceInstructionKind::LHU => expand_lhu(&instruction),
        SourceInstructionKind::LW => expand_lw(&instruction),
        SourceInstructionKind::LWU => expand_lwu(&instruction),
        SourceInstructionKind::AdviceLB => expand_advice_lb(&instruction),
        SourceInstructionKind::AdviceLH => expand_advice_lh(&instruction),
        SourceInstructionKind::AdviceLW => expand_advice_lw(&instruction),
        SourceInstructionKind::AdviceLD => expand_advice_ld(&instruction),
        SourceInstructionKind::AMOADDD => expand_amoaddd(&instruction),
        SourceInstructionKind::AMOANDD => expand_amoandd(&instruction),
        SourceInstructionKind::AMOORD => expand_amoord(&instruction),
        SourceInstructionKind::AMOXORD => expand_amoxord(&instruction),
        SourceInstructionKind::AMOSWAPD => expand_amoswapd(&instruction),
        SourceInstructionKind::AMOMAXD => expand_amomaxd(&instruction),
        SourceInstructionKind::AMOMAXUD => expand_amomaxud(&instruction),
        SourceInstructionKind::AMOMIND => expand_amomind(&instruction),
        SourceInstructionKind::AMOMINUD => expand_amominud(&instruction),
        SourceInstructionKind::AMOADDW => expand_amoaddw(&instruction),
        SourceInstructionKind::AMOANDW => expand_amoandw(&instruction),
        SourceInstructionKind::AMOORW => expand_amoorw(&instruction),
        SourceInstructionKind::AMOXORW => expand_amoxorw(&instruction),
        SourceInstructionKind::AMOSWAPW => expand_amoswapw(&instruction),
        SourceInstructionKind::AMOMAXW => expand_amomaxw(&instruction),
        SourceInstructionKind::AMOMAXUW => expand_amomaxuw(&instruction),
        SourceInstructionKind::AMOMINW => expand_amominw(&instruction),
        SourceInstructionKind::AMOMINUW => expand_amominuw(&instruction),
        SourceInstructionKind::LRD => expand_lrd(&instruction),
        SourceInstructionKind::LRW => expand_lrw(&instruction),
        SourceInstructionKind::DIV => expand_div(&instruction),
        SourceInstructionKind::DIVU => expand_divu(&instruction),
        SourceInstructionKind::DIVW => expand_divw(&instruction),
        SourceInstructionKind::DIVUW => expand_divuw(&instruction),
        SourceInstructionKind::REM => expand_rem(&instruction),
        SourceInstructionKind::REMU => expand_remu(&instruction),
        SourceInstructionKind::REMW => expand_remw(&instruction),
        SourceInstructionKind::REMUW => expand_remuw(&instruction),
        SourceInstructionKind::SB => expand_sb(&instruction),
        SourceInstructionKind::SCD => expand_scd(&instruction),
        SourceInstructionKind::SCW => expand_scw(&instruction),
        SourceInstructionKind::SH => expand_sh(&instruction),
        SourceInstructionKind::SW => expand_sw(&instruction),
        SourceInstructionKind::CSRRW => expand_csrrw(&instruction),
        SourceInstructionKind::CSRRS => expand_csrrs(&instruction),
        SourceInstructionKind::EBREAK => expand_ebreak(&instruction),
        SourceInstructionKind::ECALL => expand_ecall(&instruction),
        SourceInstructionKind::MRET => expand_mret(&instruction),
        SourceInstructionKind::SLL => expand_sll(&instruction),
        SourceInstructionKind::SLLI => expand_slli(&instruction),
        SourceInstructionKind::SLLW => expand_sllw(&instruction),
        SourceInstructionKind::SLLIW => expand_slliw(&instruction),
        SourceInstructionKind::SRL => expand_srl(&instruction),
        SourceInstructionKind::SRLI => expand_srli(&instruction),
        SourceInstructionKind::SRA => expand_sra(&instruction),
        SourceInstructionKind::SRAI => expand_srai(&instruction),
        SourceInstructionKind::SRLIW => expand_srliw(&instruction),
        SourceInstructionKind::SRAIW => expand_sraiw(&instruction),
        SourceInstructionKind::SRLW => expand_srlw(&instruction),
        SourceInstructionKind::SRAW => expand_sraw(&instruction),
        _ => Err(ExpansionError::UnsupportedInstruction),
    }
}

pub fn expand_program(
    instructions: &[SourceInstruction],
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_program_with_provider(instructions, &mut NoInlineExpansionProvider)
}

pub fn expand_program_with_provider<P: InlineExpansionProvider + ?Sized>(
    instructions: &[SourceInstruction],
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
