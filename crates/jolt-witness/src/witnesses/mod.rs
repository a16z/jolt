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

use jolt_field::JoltField;
use jolt_lookup_tables::{JoltLookupQuery, LookupQuery};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::{
    CircuitFlags, JoltCycle, JoltInstruction, JoltInstructionRow, JoltTraceRow as TraceRow,
    NormalizedOperands,
};

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
pub use pc::{BytecodePc, NextPc, NextUnexpandedPc, Pc, UnexpandedPc};
pub use ram::{RamAddress, RamHammingWeight, RamReadValue, RamWriteValue, RemappedRamAddress};
pub use registers::{RdWriteValue, Rs1Value, Rs2Value};

pub(crate) use ram::ram_access_address;

/// Non-row inputs of witness extraction: the preprocessing (bytecode PC
/// mapping, memory layout). Constructed by backends; opaque to consumers.
pub struct WitnessEnv<'a> {
    pub(crate) preprocessing: &'a JoltProgramPreprocessing,
}

impl<'a> WitnessEnv<'a> {
    pub fn new(preprocessing: &'a JoltProgramPreprocessing) -> Self {
        Self { preprocessing }
    }
}

/// The field encoding of an atomic witness value.
pub trait ToField {
    fn to_field<F: JoltField>(self) -> F;
}

/// The single-sourced derivation of one atomic witness from a trace row.
pub trait Extract<R = TraceRow>: Sized {
    fn extract(row: &R, next: Option<&R>, env: &WitnessEnv<'_>) -> Result<Self, WitnessError>;
}

/// [`Extract`] for indexed witness families ([`OpFlag`], [`InstructionFlag`],
/// [`LookupTableFlag`]): which member is extracted is bound at the use site.
pub trait ExtractIndexed<I, R = TraceRow>: Sized {
    fn extract_indexed(
        index: I,
        row: &R,
        next: Option<&R>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

fn instruction_row(row: &TraceRow) -> JoltInstructionRow {
    let circuit_flags = row.circuit_flags();
    JoltInstructionRow {
        instruction_kind: row.instruction_kind().unwrap_or_default(),
        address: row.unexpanded_pc() as usize,
        operands: NormalizedOperands {
            rs1: row.rs1_index(),
            rs2: row.rs2_index(),
            rd: row.rd_index(),
            imm: row.imm(),
        },
        virtual_sequence_remaining: circuit_flags[CircuitFlags::VirtualInstruction]
            .then_some(u16::from(!circuit_flags[CircuitFlags::IsLastInSequence])),
        is_first_in_sequence: circuit_flags[CircuitFlags::IsFirstInSequence],
        is_compressed: circuit_flags[CircuitFlags::IsCompressed],
    }
}

/// Proof-only view of the compact trace row used by lookup queries.
#[derive(Clone, Copy)]
pub(crate) struct CompactTraceCycle<'a>(&'a TraceRow);

impl JoltCycle for CompactTraceCycle<'_> {
    type Instruction = JoltInstructionRow;

    #[inline(always)]
    fn instruction(&self) -> Self::Instruction {
        instruction_row(self.0)
    }

    #[inline(always)]
    fn rs1_val(&self) -> Option<u64> {
        self.0.rs1_index().map(|_| self.0.rs1_value())
    }

    #[inline(always)]
    fn rs2_val(&self) -> Option<u64> {
        self.0.rs2_index().map(|_| self.0.rs2_value())
    }

    #[inline(always)]
    fn rd_vals(&self) -> Option<(u64, u64)> {
        self.0
            .rd_index()
            .map(|_| (self.0.rd_pre_value(), self.0.rd_write_value()))
    }

    #[inline(always)]
    fn ram_access_address(&self) -> Option<u64> {
        (self.0.is_load() || self.0.is_store()).then(|| self.0.ram_address())
    }

    #[inline(always)]
    fn ram_read_value(&self) -> Option<u64> {
        (self.0.is_load() || self.0.is_store()).then(|| self.0.ram_read_value())
    }

    #[inline(always)]
    fn ram_write_value(&self) -> Option<u64> {
        self.0.is_store().then(|| self.0.ram_write_value())
    }
}

pub(crate) fn lookup_query(row: &TraceRow) -> JoltLookupQuery<CompactTraceCycle<'_>> {
    JoltLookupQuery::new(
        row.instruction_kind().unwrap_or_default(),
        CompactTraceCycle(row),
    )
}

macro_rules! define_lookup_values {
    (
        instructions: [$($(#[$meta:meta])* $kind:ident => $variant:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        #[inline]
        pub fn lookup_values(row: &TraceRow) -> ((u64, i128), (u64, u128), u64) {
            let cycle = CompactTraceCycle(row);
            match row.instruction_kind().unwrap_or_default() {
                JoltInstruction::Noop(_) => ((0, 0), (0, 0), 0),
                $(
                    $(#[$meta])*
                    JoltInstruction::$variant(_) => {
                        let instruction = jolt_riscv::instructions::$variant(cycle);
                        (
                            LookupQuery::<{ crate::RV64_XLEN }>::to_instruction_inputs(&instruction),
                            LookupQuery::<{ crate::RV64_XLEN }>::to_lookup_operands(&instruction),
                            LookupQuery::<{ crate::RV64_XLEN }>::to_lookup_output(&instruction),
                        )
                    }
                )*
            }
        }
    };
}

jolt_riscv::for_each_jolt_instruction_kind!(define_lookup_values);

pub(crate) fn decode_instruction(row: &TraceRow) -> Result<JoltInstruction, WitnessError> {
    JoltInstruction::try_from(instruction_row(row)).map_err(|kind| {
        WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!("unsupported Jolt instruction kind in trace row: {kind:?}"),
        }
    })
}

pub(crate) fn row_is_noop(row: &TraceRow) -> bool {
    row.is_noop()
}
