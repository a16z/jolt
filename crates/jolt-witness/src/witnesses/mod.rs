//! Atomic witness values: one newtype per witness, each with its
//! single-sourced trace derivation.
//!
//! Every file holds a family's newtypes together with their [`Extract`]
//! impls — the value type, its field encoding, and its derivation from a
//! trace row live side by side, and every consumer path (oracle tables,
//! bundles, streams) dispatches to the same impl. The newtypes themselves
//! are plain values: a backend with a different row representation can
//! construct them directly. File grouping here is packaging convenience,
//! not taxonomy — nothing dispatches on modules.
//!
//! Extractors recompute from row accessors — no memoization. The two
//! irreducible non-row inputs are the lookahead window (the `Next*` family
//! is a function of rows `t` and `t + 1`, with padding semantics at
//! `T - 1`) and the environment ([`WitnessEnv`]).

use jolt_lookup_tables::JoltLookupQuery;
use jolt_program::{execution::TraceRow, preprocess::JoltProgramPreprocessing};
use jolt_riscv::{Flags, JoltInstruction, JoltInstructionKind};

use crate::protocols::jolt_vm::JOLT_VM_NAMESPACE;
use crate::WitnessError;

mod flags;
mod lookups;
mod operands;
mod pc;
mod ram;
mod registers;

pub use flags::{
    InstructionFlag, InstructionRafFlag, LookupTableFlag, NextIsFirstInSequence, NextIsNoop,
    NextIsVirtual, OpFlag, ShouldBranch, ShouldJump,
};
pub use lookups::LookupOutput;
pub use operands::{
    Imm, LeftInstructionInput, LeftLookupOperand, Product, RightInstructionInput,
    RightLookupOperand,
};
pub use pc::{NextPc, NextUnexpandedPc, Pc, UnexpandedPc};
pub use ram::{RamAddress, RamHammingWeight, RamReadValue, RamWriteValue};
pub use registers::{RdWriteValue, Rs1Value, Rs2Value};

/// Non-row inputs of witness extraction: the preprocessing (bytecode PC
/// mapping, memory layout).
pub(crate) struct WitnessEnv<'a> {
    pub(crate) preprocessing: &'a JoltProgramPreprocessing,
}

/// The single-sourced derivation of one atomic witness from a trace row.
pub(crate) trait Extract: Sized {
    fn extract(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

/// [`Extract`] for indexed witness families ([`OpFlag`], [`InstructionFlag`],
/// [`LookupTableFlag`]): which member is extracted is bound at the use site.
pub(crate) trait ExtractIndexed<I>: Sized {
    fn extract_indexed(
        index: I,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

pub(crate) fn lookup_query(row: &TraceRow) -> JoltLookupQuery<&TraceRow> {
    JoltLookupQuery::new(row.instruction.instruction_kind, row)
}

pub(crate) fn decode_instruction(row: &TraceRow) -> Result<JoltInstruction, WitnessError> {
    JoltInstruction::try_from(row.instruction).map_err(|kind| WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
    })
}

pub(crate) fn row_circuit_flags(
    row: &TraceRow,
) -> Result<jolt_riscv::CircuitFlagSet, WitnessError> {
    Ok(decode_instruction(row)?.circuit_flags())
}

pub(crate) fn row_is_noop(row: &TraceRow) -> bool {
    row.instruction.instruction_kind == JoltInstructionKind::NoOp
}

pub(crate) fn pc_for_row(
    row: &TraceRow,
    preprocessing: &JoltProgramPreprocessing,
) -> Result<usize, WitnessError> {
    preprocessing
        .bytecode
        .get_pc(&row.instruction)
        .ok_or_else(|| missing_pc_mapping(row))
}

pub(crate) fn missing_pc_mapping(row: &TraceRow) -> WitnessError {
    WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!(
            "bytecode preprocessing is missing PC mapping for address {:#x} with virtual_sequence_remaining {:?}",
            row.instruction.address, row.instruction.virtual_sequence_remaining
        ),
    }
}
