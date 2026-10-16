use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_field::JoltField;
#[cfg(feature = "parallel")]
use jolt_utils::FirstErrorLatch;
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
#[cfg(feature = "parallel")]
use jolt_witness::TraceRows;
use jolt_witness::{
    stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle, WitnessError,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::sparse::{SeedEntry, Seeds};
use crate::KernelError;

/// Operand indices and raw values for one cycle.
/// Manual because atomic witness types do not expose operand indices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RegisterCycleRow {
    /// `(register, read value)`.
    pub rs1: Option<(u8, u64)>,
    /// `(register, read value)`.
    pub rs2: Option<(u8, u64)>,
    /// `(register, pre-write value, post-write value)`.
    pub rd: Option<(u8, u64, u64)>,
}

impl WitnessBundle for RegisterCycleRow {
    // The hidden re-export avoids a jolt-program dependency.
    fn from_row(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let cycle = Self {
            rs1: row.rs1_index().map(|register| (register, row.rs1_value())),
            rs2: row.rs2_index().map(|register| (register, row.rs2_value())),
            rd: row
                .rd_index()
                .map(|register| (register, row.rd_pre_value(), row.rd_write_value())),
        };
        // Match the trace oracle's register-domain check.
        for register in [
            cycle.rs1.map(|(register, _)| register),
            cycle.rs2.map(|(register, _)| register),
            cycle.rd.map(|(register, ..)| register),
        ]
        .into_iter()
        .flatten()
        {
            if usize::from(register) >= 1usize << REGISTER_ADDRESS_BITS {
                return Err(WitnessError::InvalidWitnessData {
                    label: "jolt_vm",
                    reason: format!(
                        "register index {register} exceeds {REGISTER_ADDRESS_BITS}-bit register read-write domain"
                    ),
                });
            }
        }
        Ok(cycle)
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

const COLLECT_CHUNK: usize = 1 << 16;

/// Signed rd write delta, or zero without an rd operand.
#[inline]
pub(super) fn raw_rd_inc(cycle: &RegisterCycleRow) -> i128 {
    match cycle.rd {
        Some((_, pre, post)) => post as i128 - pre as i128,
        None => 0,
    }
}

/// Per-cycle signed rd increments.
pub(super) struct RegisterColumns {
    pub(super) rd_inc: Vec<i128>,
}

/// The sequential pass, which also keeps every seed cell.
struct CollectRegisterEntries {
    entries: Vec<SeedEntry>,
    columns: RegisterColumns,
}

impl StreamConsumer for CollectRegisterEntries {
    type Witness = RegisterCycleRow;

    fn consume(&mut self, chunk: &[RegisterCycleRow]) {
        let columns = &mut self.columns;
        for cycle in chunk {
            let row = columns.rd_inc.len() as u32;
            let (cells, len) = cycle.entries(row);
            self.entries.extend_from_slice(&cells[..len]);
            columns.rd_inc.push(raw_rd_inc(cycle));
        }
    }
}

impl RegisterColumns {
    /// Extracts every cycle once, failing on the first invalid row. Seed
    /// cells are kept only when the witness cannot serve its rows again.
    pub(super) fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<(Self, Seeds), KernelError<F>> {
        if u32::try_from(cycles.saturating_sub(1)).is_err() {
            return Err(KernelError::InvariantViolation {
                reason: "cycle count exceeds the seed entries' packed u32 row domain",
            });
        }
        #[cfg(feature = "parallel")]
        if let Some(access) = witness.rows() {
            if cycles <= access.cycles() {
                return Ok((Self::collect_par(&access, cycles)?, Seeds::Rows(access)));
            }
        }
        let mut consumers = (CollectRegisterEntries {
            entries: Vec::with_capacity(cycles * 3),
            columns: Self {
                rd_inc: Vec::with_capacity(cycles),
            },
        },);
        stream_witnesses(witness, 0..cycles, COLLECT_CHUNK, &mut consumers)?;
        let CollectRegisterEntries { entries, columns } = consumers.0;
        Ok((columns, Seeds::Collected(entries)))
    }

    #[cfg(feature = "parallel")]
    fn collect_par<F: JoltField>(
        access: &TraceRows,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        const CHUNK: usize = 1 << 14;
        let mut rd_inc: Vec<i128> = Vec::with_capacity(cycles);
        let error = FirstErrorLatch::new();
        rd_inc.spare_capacity_mut()[..cycles]
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_index, inc)| {
                let base = chunk_index * CHUNK;
                for (offset, slot) in inc.iter_mut().enumerate() {
                    match access.window::<RegisterCycleRow>(base + offset) {
                        Ok(cycle) => {
                            let _ = slot.write(raw_rd_inc(&cycle));
                        }
                        Err(failure) => {
                            error.record(base + offset, failure);
                            return;
                        }
                    }
                }
            });
        if let Some(failure) = error.take() {
            return Err(failure.into());
        }
        // SAFETY: the error latch is empty, so every chunk ran to completion
        // and initialized its whole span of the column.
        unsafe {
            rd_inc.set_len(cycles);
        }
        Ok(Self { rd_inc })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{
        JoltInstructionKind, JoltInstructionRow, JoltTraceRow, NormalizedOperands, RamAccess,
        RegisterState, RV64IMAC_JOLT,
    };

    use super::*;

    #[test]
    fn rejects_register_outside_protocol_domain() {
        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rs1: Some(200),
                ..Default::default()
            },
            ..Default::default()
        };
        let bytecode = BytecodePreprocessing::preprocess(
            vec![instruction],
            instruction.address as u64,
            RV64IMAC_JOLT,
        )
        .unwrap();
        let pc = u32::try_from(bytecode.get_pc(&instruction).unwrap()).unwrap();
        let row =
            JoltTraceRow::new(instruction, RegisterState::default(), RamAccess::NoOp, pc).unwrap();
        let preprocessing = JoltProgramPreprocessing {
            bytecode,
            ram: RAMPreprocessing::default(),
            memory_layout: MemoryLayout::default(),
            max_padded_trace_length: 1,
        };
        let env = WitnessEnv::new(&preprocessing);

        let error = RegisterCycleRow::from_row(&row, None, &env).unwrap_err();
        assert!(matches!(
            error,
            WitnessError::InvalidWitnessData { reason, .. }
                if reason.contains("register index 200")
        ));
    }
}
