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
mod inline;
mod materialize;
mod memory;
mod metadata;
mod operands;
mod shifts;
#[cfg(test)]
mod tests;

pub use allocator::ExpansionAllocator;
pub use error::ExpansionError;
pub use grammar::ExpandedInstructionSequence;
pub use inline::{
    InlineExpansionBuilder, InlineInstruction, InlineOperands, InlineRegister, Value,
};

use allocator::{
    mcause_register, mepc_register, mstatus_register, mtval_register, reservation_d_register,
    reservation_w_register, trap_handler_register, virtual_register_for_csr,
};
use arithmetic::*;
use control_flow::*;
use division::*;
use grammar::{reg, ExpansionBuilder, RegisterOperand, TempId};
use jolt_riscv::{
    JoltInstruction, JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow,
    NormalizedOperands, SourceInstruction, SourceInstructionKind, SourceInstructionRow,
};
use materialize::ExpansionState;
use memory::*;
use operands::*;
use shifts::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyRequirement {
    FieldCanonicalOutput,
    NonZeroDivisor,
    ModularRelation,
    GlvSignWords,
    DecompositionRecomposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineAdmissibility {
    Public {
        requirements: &'static [SafetyRequirement],
    },
    InternalOnly {
        reason: &'static str,
    },
}

pub trait InlineExpansionProvider {
    /// Builds a registered inline row's symbolic expansion recipe.
    ///
    /// The top-level entry point remaps `rd = x0` before calling this hook,
    /// then materializes the recipe, appends inline reset rows, validates
    /// target legality, and stamps sequence metadata.
    fn expand_inline(
        &mut self,
        instruction: &SourceInstruction,
        profile: JoltInstructionProfile,
    ) -> Result<ExpandedInstructionSequence, ExpansionError>;
}

#[derive(Debug, Default)]
pub struct NoInlineExpansionProvider;

impl InlineExpansionProvider for NoInlineExpansionProvider {
    fn expand_inline(
        &mut self,
        _instruction: &SourceInstruction,
        _profile: JoltInstructionProfile,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Err(ExpansionError::InlineProviderRequired)
    }
}

pub fn expand_instruction(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    expand_instruction_with_provider(
        instruction,
        allocator,
        &mut NoInlineExpansionProvider,
        profile,
    )
}

pub fn expand_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    expand_source_instruction_with_provider(instruction, allocator, inline_provider, profile)
}

fn expand_source_instruction_with_provider<P: InlineExpansionProvider + ?Sized>(
    instruction: &SourceInstruction,
    allocator: &mut ExpansionAllocator,
    inline_provider: &mut P,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    let rewritten_source;
    let mut allocated_rd_zero_register = None;
    let instruction = if instruction.row().operands.rd == Some(0)
        && !handles_rd_zero_internally(instruction.kind())
    {
        if instruction.kind().has_side_effects() {
            let virtual_register = allocator.allocate()?;
            allocated_rd_zero_register = Some(virtual_register);
            rewritten_source = (*instruction).map_row(|mut row| {
                row.operands.rd = Some(virtual_register);
                row
            });
            &rewritten_source
        } else {
            return final_rows_to_instructions(vec![noop_for(*instruction.row())], profile);
        }
    } else {
        instruction
    };
    let result = if instruction.kind() == SourceInstructionKind::Inline {
        let owned_allocator = std::mem::take(allocator);
        let mut state = ExpansionState::new(owned_allocator, profile);
        let result = inline_provider
            .expand_inline(instruction, profile)
            .and_then(|sequence| state.materialize_inline(sequence))
            .and_then(|rows| final_rows_to_instructions(rows, profile));
        *allocator = state.into_allocator();
        result
    } else {
        let owned_allocator = std::mem::take(allocator);
        let mut state = ExpansionState::new(owned_allocator, profile);
        let result = state
            .expand_source_recursive(instruction)
            .and_then(|rows| final_rows_to_instructions(rows, profile));
        *allocator = state.into_allocator();
        result
    };
    if let Some(register) = allocated_rd_zero_register {
        allocator.release(register)?;
    }
    result
}

fn final_rows_to_instructions(
    rows: Vec<JoltInstructionRow>,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    rows.into_iter()
        .map(|row| {
            if !profile.supports_jolt(row.instruction_kind) {
                return Err(ExpansionError::IllegalTargetInstruction(
                    row.instruction_kind,
                ));
            }
            JoltInstruction::try_from(row).map_err(ExpansionError::IllegalTargetInstruction)
        })
        .collect()
}

/// Dispatches one source-only instruction to the recipe that explains its
/// final bytecode semantics.
///
/// Each callee returns a symbolic sequence, not concrete rows. During
/// materialization, `emit_*` rows become final bytecode directly while
/// `expand_*` rows are routed back through this dispatcher. That recursive
/// route is intentional: common substeps such as narrow loads, word shifts, and
/// virtual assertions keep one definition of their own lowering contract.
fn expand_source_only_instruction(
    instruction: &SourceInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let row = instruction.row();
    match instruction.kind() {
        SourceInstructionKind::ADDIW => expand_addiw(row),
        SourceInstructionKind::ADDW => expand_addw(row),
        SourceInstructionKind::SUBW => expand_subw(row),
        SourceInstructionKind::MULH => expand_mulh(row),
        SourceInstructionKind::MULHSU => expand_mulhsu(row),
        SourceInstructionKind::MULW => expand_mulw(row),
        SourceInstructionKind::LB => expand_lb(row),
        SourceInstructionKind::LBU => expand_lbu(row),
        SourceInstructionKind::LH => expand_lh(row),
        SourceInstructionKind::LHU => expand_lhu(row),
        SourceInstructionKind::LW => expand_lw(row),
        SourceInstructionKind::LWU => expand_lwu(row),
        SourceInstructionKind::AdviceLB => expand_advice_lb(row),
        SourceInstructionKind::AdviceLH => expand_advice_lh(row),
        SourceInstructionKind::AdviceLW => expand_advice_lw(row),
        SourceInstructionKind::AdviceLD => expand_advice_ld(row),
        SourceInstructionKind::AMOADDD => expand_amoaddd(row),
        SourceInstructionKind::AMOANDD => expand_amoandd(row),
        SourceInstructionKind::AMOORD => expand_amoord(row),
        SourceInstructionKind::AMOXORD => expand_amoxord(row),
        SourceInstructionKind::AMOSWAPD => expand_amoswapd(row),
        SourceInstructionKind::AMOMAXD => expand_amomaxd(row),
        SourceInstructionKind::AMOMAXUD => expand_amomaxud(row),
        SourceInstructionKind::AMOMIND => expand_amomind(row),
        SourceInstructionKind::AMOMINUD => expand_amominud(row),
        SourceInstructionKind::AMOADDW => expand_amoaddw(row),
        SourceInstructionKind::AMOANDW => expand_amoandw(row),
        SourceInstructionKind::AMOORW => expand_amoorw(row),
        SourceInstructionKind::AMOXORW => expand_amoxorw(row),
        SourceInstructionKind::AMOSWAPW => expand_amoswapw(row),
        SourceInstructionKind::AMOMAXW => expand_amomaxw(row),
        SourceInstructionKind::AMOMAXUW => expand_amomaxuw(row),
        SourceInstructionKind::AMOMINW => expand_amominw(row),
        SourceInstructionKind::AMOMINUW => expand_amominuw(row),
        SourceInstructionKind::LRD => expand_lrd(row),
        SourceInstructionKind::LRW => expand_lrw(row),
        SourceInstructionKind::DIV => expand_div(row),
        SourceInstructionKind::DIVU => expand_divu(row),
        SourceInstructionKind::DIVW => expand_divw(row),
        SourceInstructionKind::DIVUW => expand_divuw(row),
        SourceInstructionKind::REM => expand_rem(row),
        SourceInstructionKind::REMU => expand_remu(row),
        SourceInstructionKind::REMW => expand_remw(row),
        SourceInstructionKind::REMUW => expand_remuw(row),
        SourceInstructionKind::SB => expand_sb(row),
        SourceInstructionKind::SCD => expand_scd(row),
        SourceInstructionKind::SCW => expand_scw(row),
        SourceInstructionKind::SH => expand_sh(row),
        SourceInstructionKind::SW => expand_sw(row),
        SourceInstructionKind::CSRRW => expand_csrrw(row),
        SourceInstructionKind::CSRRS => expand_csrrs(row),
        SourceInstructionKind::EBREAK => expand_ebreak(row),
        SourceInstructionKind::ECALL => expand_ecall(row),
        SourceInstructionKind::MRET => expand_mret(row),
        SourceInstructionKind::SLL => expand_sll(row),
        SourceInstructionKind::SLLI => expand_slli(row),
        SourceInstructionKind::SLLW => expand_sllw(row),
        SourceInstructionKind::SLLIW => expand_slliw(row),
        SourceInstructionKind::SRL => expand_srl(row),
        SourceInstructionKind::SRLI => expand_srli(row),
        SourceInstructionKind::SRA => expand_sra(row),
        SourceInstructionKind::SRAI => expand_srai(row),
        SourceInstructionKind::SRLIW => expand_srliw(row),
        SourceInstructionKind::SRAIW => expand_sraiw(row),
        SourceInstructionKind::SRLW => expand_srlw(row),
        SourceInstructionKind::SRAW => expand_sraw(row),
        _ => Err(ExpansionError::UnsupportedInstruction),
    }
}

pub fn expand_program(
    instructions: &[SourceInstruction],
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    expand_program_with_provider(instructions, &mut NoInlineExpansionProvider, profile)
}

pub fn expand_program_with_provider<P: InlineExpansionProvider + ?Sized>(
    instructions: &[SourceInstruction],
    inline_provider: &mut P,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstruction>, ExpansionError> {
    let mut allocator = ExpansionAllocator::new();
    let mut expanded = Vec::new();
    for instruction in instructions {
        expanded.extend(expand_source_instruction_with_provider(
            instruction,
            &mut allocator,
            inline_provider,
            profile,
        )?);
    }
    Ok(expanded)
}
