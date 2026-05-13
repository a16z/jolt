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
    JoltInstruction, JoltInstructionKind, JoltRow, NormalizedOperands, SourceInstruction,
    SourceInstructionKind, RV64IMAC_JOLT,
};
use materialize::ExpansionState;
use memory::*;
use metadata::stamp_inline_sequence;
use operands::*;
use shifts::*;

pub trait InlineExpansionProvider {
    /// Expands a registered inline row into final Jolt instructions.
    ///
    /// Provider output intentionally stays outside the provider-free builder
    /// core. The top-level entry point remaps `rd = x0` before calling this
    /// hook, then validates target legality and stamps sequence metadata.
    fn expand_inline(
        &mut self,
        instruction: &SourceInstruction,
        allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction>, ExpansionError>;
}

#[derive(Debug, Default)]
pub struct NoInlineExpansionProvider;

impl InlineExpansionProvider for NoInlineExpansionProvider {
    fn expand_inline(
        &mut self,
        _instruction: &SourceInstruction,
        _allocator: &mut ExpansionAllocator,
    ) -> Result<Vec<JoltInstruction>, ExpansionError> {
        Err(ExpansionError::InlineProviderRequired)
    }
}

pub fn expand_instruction(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    expand_instruction_with_provider(instruction, allocator, &mut NoInlineExpansionProvider)
}

pub fn expand_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    expand_source_instruction_with_provider(instruction, allocator, inline_provider)
}

fn expand_source_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    let rewritten_source;
    let mut allocated_rd_zero_register = None;
    let instruction = if instruction.row().operands.rd == Some(0)
        && !handles_rd_zero_internally(instruction.kind())
    {
        if instruction.kind().jolt_kind().has_side_effects() {
            let virtual_register = allocator.allocate()?;
            allocated_rd_zero_register = Some(virtual_register);
            rewritten_source = (*instruction).map_row(|mut row| {
                row.operands.rd = Some(virtual_register);
                row
            });
            &rewritten_source
        } else {
            return final_rows_to_instructions(vec![noop_for(instruction.jolt_row())]);
        }
    } else {
        instruction
    };
    let jolt_instruction = instruction.jolt_row();

    let result = if instruction.kind() == SourceInstructionKind::Inline {
        let instructions = inline_provider.expand_inline(instruction, allocator)?;
        finalize_inline_provider_instructions(jolt_instruction, allocator, instructions)
    } else {
        let owned_allocator = std::mem::take(allocator);
        let mut state = ExpansionState::new(owned_allocator);
        let result = state
            .expand_source_recursive(instruction)
            .and_then(final_rows_to_instructions);
        *allocator = state.into_allocator();
        result
    };
    if let Some(register) = allocated_rd_zero_register {
        allocator.release(register)?;
    }
    result
}

fn finalize_inline_provider_instructions(
    source: JoltRow,
    allocator: &mut ExpansionAllocator,
    instructions: Vec<JoltInstruction>,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    let mut rows = instructions
        .into_iter()
        .map(JoltRow::from)
        .collect::<Vec<_>>();
    for register in allocator.take_registers_for_reset()? {
        rows.push(JoltRow {
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
    final_rows_to_instructions(stamp_inline_sequence(rows, source.is_compressed)?)
}

fn final_rows_to_instructions(rows: Vec<JoltRow>) -> Result<Vec<JoltInstruction>, ExpansionError> {
    rows.into_iter()
        .map(|row| {
            if !RV64IMAC_JOLT.supports_jolt(row.instruction_kind) {
                return Err(ExpansionError::IllegalTargetInstruction(
                    row.instruction_kind,
                ));
            }
            JoltInstruction::try_from(row).map_err(ExpansionError::IllegalTargetInstruction)
        })
        .collect()
}

/// Dispatches a source-only instruction to its recipe builder (phase 1).
fn expand_source_only_instruction(
    instruction: &SourceInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let row = instruction.jolt_row();
    match instruction.kind() {
        SourceInstructionKind::ADDIW => expand_addiw(&row),
        SourceInstructionKind::ADDW => expand_addw(&row),
        SourceInstructionKind::SUBW => expand_subw(&row),
        SourceInstructionKind::MULH => expand_mulh(&row),
        SourceInstructionKind::MULHSU => expand_mulhsu(&row),
        SourceInstructionKind::MULW => expand_mulw(&row),
        SourceInstructionKind::LB => expand_lb(&row),
        SourceInstructionKind::LBU => expand_lbu(&row),
        SourceInstructionKind::LH => expand_lh(&row),
        SourceInstructionKind::LHU => expand_lhu(&row),
        SourceInstructionKind::LW => expand_lw(&row),
        SourceInstructionKind::LWU => expand_lwu(&row),
        SourceInstructionKind::AdviceLB => expand_advice_lb(&row),
        SourceInstructionKind::AdviceLH => expand_advice_lh(&row),
        SourceInstructionKind::AdviceLW => expand_advice_lw(&row),
        SourceInstructionKind::AdviceLD => expand_advice_ld(&row),
        SourceInstructionKind::AMOADDD => expand_amoaddd(&row),
        SourceInstructionKind::AMOANDD => expand_amoandd(&row),
        SourceInstructionKind::AMOORD => expand_amoord(&row),
        SourceInstructionKind::AMOXORD => expand_amoxord(&row),
        SourceInstructionKind::AMOSWAPD => expand_amoswapd(&row),
        SourceInstructionKind::AMOMAXD => expand_amomaxd(&row),
        SourceInstructionKind::AMOMAXUD => expand_amomaxud(&row),
        SourceInstructionKind::AMOMIND => expand_amomind(&row),
        SourceInstructionKind::AMOMINUD => expand_amominud(&row),
        SourceInstructionKind::AMOADDW => expand_amoaddw(&row),
        SourceInstructionKind::AMOANDW => expand_amoandw(&row),
        SourceInstructionKind::AMOORW => expand_amoorw(&row),
        SourceInstructionKind::AMOXORW => expand_amoxorw(&row),
        SourceInstructionKind::AMOSWAPW => expand_amoswapw(&row),
        SourceInstructionKind::AMOMAXW => expand_amomaxw(&row),
        SourceInstructionKind::AMOMAXUW => expand_amomaxuw(&row),
        SourceInstructionKind::AMOMINW => expand_amominw(&row),
        SourceInstructionKind::AMOMINUW => expand_amominuw(&row),
        SourceInstructionKind::LRD => expand_lrd(&row),
        SourceInstructionKind::LRW => expand_lrw(&row),
        SourceInstructionKind::DIV => expand_div(&row),
        SourceInstructionKind::DIVU => expand_divu(&row),
        SourceInstructionKind::DIVW => expand_divw(&row),
        SourceInstructionKind::DIVUW => expand_divuw(&row),
        SourceInstructionKind::REM => expand_rem(&row),
        SourceInstructionKind::REMU => expand_remu(&row),
        SourceInstructionKind::REMW => expand_remw(&row),
        SourceInstructionKind::REMUW => expand_remuw(&row),
        SourceInstructionKind::SB => expand_sb(&row),
        SourceInstructionKind::SCD => expand_scd(&row),
        SourceInstructionKind::SCW => expand_scw(&row),
        SourceInstructionKind::SH => expand_sh(&row),
        SourceInstructionKind::SW => expand_sw(&row),
        SourceInstructionKind::CSRRW => expand_csrrw(&row),
        SourceInstructionKind::CSRRS => expand_csrrs(&row),
        SourceInstructionKind::EBREAK => expand_ebreak(&row),
        SourceInstructionKind::ECALL => expand_ecall(&row),
        SourceInstructionKind::MRET => expand_mret(&row),
        SourceInstructionKind::SLL => expand_sll(&row),
        SourceInstructionKind::SLLI => expand_slli(&row),
        SourceInstructionKind::SLLW => expand_sllw(&row),
        SourceInstructionKind::SLLIW => expand_slliw(&row),
        SourceInstructionKind::SRL => expand_srl(&row),
        SourceInstructionKind::SRLI => expand_srli(&row),
        SourceInstructionKind::SRA => expand_sra(&row),
        SourceInstructionKind::SRAI => expand_srai(&row),
        SourceInstructionKind::SRLIW => expand_srliw(&row),
        SourceInstructionKind::SRAIW => expand_sraiw(&row),
        SourceInstructionKind::SRLW => expand_srlw(&row),
        SourceInstructionKind::SRAW => expand_sraw(&row),
        _ => Err(ExpansionError::UnsupportedInstruction),
    }
}

pub fn expand_program(
    instructions: &[SourceInstruction],
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    expand_program_with_provider(instructions, &mut NoInlineExpansionProvider)
}

pub fn expand_program_with_provider<P: InlineExpansionProvider + ?Sized>(
    instructions: &[SourceInstruction],
    inline_provider: &mut P,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let mut expanded = Vec::new();
    for instruction in instructions {
        expanded.extend(expand_source_instruction_with_provider(
            instruction,
            &mut allocator,
            inline_provider,
        )?);
    }
    Ok(expanded)
}
