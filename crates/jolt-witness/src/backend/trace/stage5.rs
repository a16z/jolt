use jolt_lookup_tables::{InstructionLookupTable, LookupTableKind};
use jolt_program::execution::{TraceRow, TraceSource};

use super::lookup::instruction_lookup_index;
use jolt_riscv::{Flags, InterleavedBitsMarker, JoltInstruction};

use super::{TraceBackend, JOLT_VM_LABEL, RV64_XLEN};
use crate::WitnessError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5InstructionReadRafRow {
    pub lookup_index: u128,
    pub table_index: Option<usize>,
    pub interleaved_operands: bool,
}

pub trait JoltVmStage5InstructionReadRafRows {
    fn stage5_instruction_read_raf_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<Stage5InstructionReadRafRow>, WitnessError>;
}

#[derive(Clone, Copy, Debug)]
struct Stage5InstructionCycle {
    address: u128,
    table: Option<LookupTableKind<RV64_XLEN>>,
    interleaved_operands: bool,
}

impl<T> JoltVmStage5InstructionReadRafRows for TraceBackend<'_, T>
where
    T: TraceSource + Clone,
{
    fn stage5_instruction_read_raf_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<Stage5InstructionReadRafRow>, WitnessError> {
        if log_t != self.config.log_t {
            return Err(invalid_dimensions(format!(
                "Stage 5 instruction rows requested log_t {log_t}, witness has {}",
                self.config.log_t
            )));
        }
        let cycles = super::checked_pow2(log_t)?;
        let mut trace = self.trace.trace.clone();
        (0..cycles)
            .map(|_| {
                let row = trace.next_row();
                let cycle = stage5_instruction_cycle(row.as_ref())?;
                Ok(Stage5InstructionReadRafRow {
                    lookup_index: cycle.address,
                    table_index: cycle.table.map(|table| table.index()),
                    interleaved_operands: cycle.interleaved_operands,
                })
            })
            .collect()
    }
}

fn stage5_instruction_cycle(
    row: Option<&TraceRow>,
) -> Result<Stage5InstructionCycle, WitnessError> {
    let address = match row {
        Some(row) => instruction_lookup_index::<RV64_XLEN>(row).map_err(|error| {
            WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: error.to_string(),
            }
        })?,
        None => 0,
    };
    let instruction_row = row.map_or_else(Default::default, |row| row.instruction);
    let instruction = JoltInstruction::try_from(instruction_row).map_err(|kind| {
        invalid_data(format!(
            "unsupported Jolt instruction kind in Stage 5 instruction row: {kind:?}"
        ))
    })?;
    let flags = instruction.circuit_flags();
    Ok(Stage5InstructionCycle {
        address,
        table: instruction.lookup_table(),
        interleaved_operands: flags.is_interleaved_operands(),
    })
}

fn invalid_dimensions(reason: impl std::fmt::Display) -> WitnessError {
    WitnessError::InvalidDimensions {
        label: JOLT_VM_LABEL,
        reason: reason.to_string(),
    }
}

fn invalid_data(reason: impl std::fmt::Display) -> WitnessError {
    WitnessError::InvalidWitnessData {
        label: JOLT_VM_LABEL,
        reason: reason.to_string(),
    }
}
