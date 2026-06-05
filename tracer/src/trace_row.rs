//! Conversion from tracer [`Cycle`]s to the proof-facing [`JoltTraceRow`].
//!
//! `JoltTraceRow` lives in `jolt-riscv` and is deliberately tracer-free. This
//! module owns the one direction that needs tracer types: extracting a cycle's
//! dynamic witness values and pairing them with its final bytecode index. That
//! keeps the dependency edge `tracer -> jolt-riscv` (not the reverse).

use jolt_program::preprocess::BytecodePreprocessing;
use jolt_riscv::{JoltTraceRow, LogicalValues, SourceInstructionKind, TraceRowError};

use crate::instruction::{Cycle, RAMAccess};

/// Error raised while converting a tracer cycle into a [`JoltTraceRow`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CycleConversionError {
    /// The cycle is a source-only / structural cycle with no final Jolt row.
    #[error("cycle has no final Jolt instruction row: {0:?}")]
    SourceOnlyCycle(SourceInstructionKind),
    /// Bytecode preprocessing has no index for this cycle's instruction.
    #[error("no bytecode index for instruction at {address:#x} (virtual_sequence_remaining={virtual_sequence_remaining:?})")]
    MissingBytecodePc {
        address: u64,
        virtual_sequence_remaining: Option<u16>,
    },
    /// The bytecode index exceeds the compact `u32` trace-row storage budget.
    #[error("bytecode index {pc} exceeds u32 trace-row storage budget")]
    BytecodePcTooWide { pc: usize },
    /// The row's logical values violate the trace-row layout constraints.
    #[error(transparent)]
    Row(TraceRowError),
}

/// Extract the dynamic logical values from a tracer [`Cycle`].
fn logical_values(cycle: &Cycle) -> LogicalValues {
    let rs1 = cycle.rs1_read();
    let rs2 = cycle.rs2_read();
    let rd = cycle.rd_write();
    let (ram_read_value, ram_write_value) = match cycle.ram_access() {
        RAMAccess::Read(read) => (read.value, read.value),
        RAMAccess::Write(write) => (write.pre_value, write.post_value),
        RAMAccess::NoOp => (0, 0),
    };
    LogicalValues {
        rs1_value: rs1.map_or(0, |(_, value)| value),
        rs2_value: rs2.map_or(0, |(_, value)| value),
        rd_pre_value: rd.map_or(0, |(_, pre, _)| pre),
        rd_write_value: rd.map_or(0, |(_, _, post)| post),
        ram_address: cycle.ram_access().address() as u64,
        ram_read_value,
        ram_write_value,
        rs1_index: rs1.map(|(index, _)| index),
        rs2_index: rs2.map(|(index, _)| index),
        rd_index: rd.map(|(index, _, _)| index),
    }
}

/// Convert a single tracer cycle into a [`JoltTraceRow`].
///
/// Rejects source-only cycles, so this is the phase-boundary check: every row
/// in the materialized trace is backed by a final Jolt instruction.
pub fn cycle_to_trace_row(
    cycle: &Cycle,
    bytecode: &BytecodePreprocessing,
) -> Result<JoltTraceRow, CycleConversionError> {
    let instruction = cycle
        .instruction()
        .try_jolt_instruction_row()
        .map_err(CycleConversionError::SourceOnlyCycle)?;
    let pc = bytecode
        .get_pc(&instruction)
        .ok_or(CycleConversionError::MissingBytecodePc {
            address: instruction.address as u64,
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        })?;
    let bytecode_pc =
        u32::try_from(pc).map_err(|_| CycleConversionError::BytecodePcTooWide { pc })?;
    JoltTraceRow::from_components(&logical_values(cycle), &instruction, bytecode_pc)
        .map_err(CycleConversionError::Row)
}

/// Materialize the full proof-facing trace once from a `Vec<Cycle>`.
pub fn build_trace_rows(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
) -> Result<Vec<JoltTraceRow>, CycleConversionError> {
    trace
        .iter()
        .map(|cycle| cycle_to_trace_row(cycle, bytecode))
        .collect()
}
