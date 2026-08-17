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

use jolt_field::Field;
use jolt_lookup_tables::JoltLookupQuery;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::{JoltInstruction, JoltTraceRow as TraceRow};

use crate::WitnessError;
use crate::JOLT_VM_LABEL;

mod flags;
mod increments;
mod lookups;
mod one_hot;
mod operands;
mod pc;
mod ram;
mod registers;
mod spartan;

pub use flags::{
    InstructionFlag, InstructionRafFlag, LookupTableFlag, NextIsFirstInSequence, NextIsNoop,
    NextIsVirtual, OpFlag, ShouldBranch, ShouldJump,
};
pub use increments::{BalancedIncColumn, BalancedIncRow, FusedInc, RamInc, RdInc};
pub use lookups::{LookupIndex, LookupOutput, TableIndex};
pub use one_hot::{BytecodeRaChunk, InstructionRaChunk, RaChunkSelector, RamRaChunk};
pub use operands::{
    Imm, LeftInstructionInput, LeftLookupOperand, Product, RightInstructionInput,
    RightLookupOperand,
};
pub use pc::{BytecodePc, MappedPc, NextPc, NextUnexpandedPc, Pc, UnexpandedPc};
pub use ram::{RamAddress, RamHammingWeight, RamReadValue, RamWriteValue, RemappedRamAddress};
pub use registers::{RdWriteValue, Rs1Value, Rs2Value};
pub use spartan::SpartanOuterRow;

pub(crate) use ram::ram_access_address;

/// Non-row inputs of witness extraction: the preprocessing (bytecode PC
/// mapping, memory layout). Constructed by backends; opaque to consumers.
pub struct WitnessEnv<'a> {
    pub(crate) preprocessing: &'a JoltProgramPreprocessing,
}

/// The field encoding of an atomic witness value.
pub trait ToField {
    fn to_field<F: Field>(self) -> F;
}

/// The single-sourced derivation of one atomic witness from a trace row.
pub trait Extract: Sized {
    fn extract(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

/// [`Extract`] for indexed witness families ([`OpFlag`], [`InstructionFlag`],
/// [`LookupTableFlag`]): which member is extracted is bound at the use site.
pub trait ExtractIndexed<I>: Sized {
    fn extract_indexed(
        index: I,
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

pub(crate) fn lookup_query(row: &TraceRow) -> JoltLookupQuery<&TraceRow> {
    JoltLookupQuery::new(row.instruction_kind().unwrap_or_default(), row)
}

pub(crate) fn decode_instruction(row: &TraceRow) -> Result<JoltInstruction, WitnessError> {
    JoltInstruction::try_from(row.instruction()).map_err(|kind| WitnessError::InvalidWitnessData {
        label: JOLT_VM_LABEL,
        reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
    })
}

pub(crate) fn row_is_noop(row: &TraceRow) -> bool {
    row.is_noop()
}
