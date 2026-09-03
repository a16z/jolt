//! Typed per-cycle register rows and the one-pass sparse-entry collection.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_field::JoltField;
#[cfg(feature = "parallel")]
use jolt_utils::FirstErrorLatch;
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
#[cfg(feature = "parallel")]
use jolt_witness::RandomAccessRows;
use jolt_witness::{
    stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle, WitnessError,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::sparse::SeedEntry;
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

/// Per-cycle `rd` indices shared with stage 5.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedRdIndices(pub Vec<Option<u8>>);

/// Row-window size for streaming collection.
const COLLECT_CHUNK: usize = 1 << 16;

/// Signed rd write delta, or zero without an rd operand.
#[inline]
fn raw_rd_inc(cycle: &RegisterCycleRow) -> i128 {
    match cycle.rd {
        Some((_, pre, post)) => post as i128 - pre as i128,
        None => 0,
    }
}

/// Builds entries, operand indices, and `rd_inc` in one trace pass.
pub(super) struct CollectRegisterEntries {
    pub(super) entries: Vec<SeedEntry>,
    pub(super) rs1_indices: Vec<Option<u8>>,
    pub(super) rs2_indices: Vec<Option<u8>>,
    pub(super) rd_indices: Vec<Option<u8>>,
    pub(super) rd_inc: Vec<i128>,
}

impl StreamConsumer for CollectRegisterEntries {
    type Witness = RegisterCycleRow;

    fn consume(&mut self, chunk: &[RegisterCycleRow]) {
        for cycle in chunk {
            let row = self.rs1_indices.len() as u32;
            let (cells, len) = cycle.entries(row);
            self.entries.extend_from_slice(&cells[..len]);
            self.rs1_indices.push(cycle.rs1.map(|(k, _)| k));
            self.rs2_indices.push(cycle.rs2.map(|(k, _)| k));
            self.rd_indices.push(cycle.rd.map(|(k, ..)| k));
            self.rd_inc.push(raw_rd_inc(cycle));
        }
    }
}

impl CollectRegisterEntries {
    /// Collects in parallel when random access is available; streams otherwise.
    pub(super) fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        // Seed rows are packed as u32.
        if u32::try_from(cycles.saturating_sub(1)).is_err() {
            return Err(KernelError::InvariantViolation {
                reason: "cycle count exceeds the seed entries' packed u32 row domain",
            });
        }
        #[cfg(feature = "parallel")]
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                return Self::collect_par(&access, cycles);
            }
        }
        let mut consumers = (CollectRegisterEntries {
            entries: Vec::with_capacity(cycles * 3),
            rs1_indices: Vec::with_capacity(cycles),
            rs2_indices: Vec::with_capacity(cycles),
            rd_indices: Vec::with_capacity(cycles),
            rd_inc: Vec::with_capacity(cycles),
        },);
        stream_witnesses(witness, 0..cycles, COLLECT_CHUNK, &mut consumers)?;
        Ok(consumers.0)
    }

    /// Two-pass parallel build: count and fill columns, then scatter entries.
    /// Exclusive offsets avoid a second entry-sized allocation.
    #[cfg(feature = "parallel")]
    fn collect_par<F: JoltField>(
        access: &RandomAccessRows,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        use core::mem::MaybeUninit;
        /// Scatter grain at about three entries per cycle.
        const CHUNK: usize = 1 << 14;
        let mut rs1_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rs2_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rd_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rd_inc: Vec<i128> = Vec::with_capacity(cycles);
        let error = FirstErrorLatch::new();
        let chunk_count = cycles.div_ceil(CHUNK);
        // Pass 1: count entries per chunk and fill the index + rd_inc columns.
        let mut counts: Vec<usize> = Vec::new();
        (
            rs1_indices.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            rs2_indices.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            rd_indices.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            rd_inc.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
        )
            .into_par_iter()
            .enumerate()
            .map(|(chunk_index, (rs1, rs2, rd, inc))| {
                let base = chunk_index * CHUNK;
                let mut count = 0usize;
                for offset in 0..rs1.len() {
                    match access.window::<RegisterCycleRow>(base + offset) {
                        Ok(cycle) => {
                            count += cycle.entry_count();
                            let _ = rs1[offset].write(cycle.rs1.map(|(k, _)| k));
                            let _ = rs2[offset].write(cycle.rs2.map(|(k, _)| k));
                            let _ = rd[offset].write(cycle.rd.map(|(k, ..)| k));
                            let _ = inc[offset].write(raw_rd_inc(&cycle));
                        }
                        Err(failure) => {
                            error.record(base + offset, failure);
                            return count;
                        }
                    }
                }
                count
            })
            .collect_into_vec(&mut counts);
        if let Some(failure) = error.take() {
            return Err(failure.into());
        }
        // SAFETY: the error latch is empty, so every chunk ran to completion
        // and initialized its whole span of all four columns.
        unsafe {
            rs1_indices.set_len(cycles);
            rs2_indices.set_len(cycles);
            rd_indices.set_len(cycles);
            rd_inc.set_len(cycles);
        }

        // Scan counts, then scatter into exclusive windows.
        let mut offsets: Vec<usize> = Vec::with_capacity(chunk_count);
        let mut total = 0usize;
        for &count in &counts {
            offsets.push(total);
            total += count;
        }
        let mut entries: Vec<SeedEntry> = Vec::with_capacity(total);
        {
            let mut rest: &mut [MaybeUninit<SeedEntry>] =
                &mut entries.spare_capacity_mut()[..total];
            let mut windows: Vec<&mut [MaybeUninit<SeedEntry>]> = Vec::with_capacity(chunk_count);
            for &count in &counts {
                let (head, tail) = rest.split_at_mut(count);
                windows.push(head);
                rest = tail;
            }
            let error = FirstErrorLatch::new();
            windows
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_index, window)| {
                    let base = chunk_index * CHUNK;
                    let top = ((chunk_index + 1) * CHUNK).min(cycles);
                    let mut written = 0usize;
                    for row in base..top {
                        // Latch unexpected second-pass extraction failures.
                        match access.window::<RegisterCycleRow>(row) {
                            Ok(cycle) => {
                                let (cells, len) = cycle.entries(row as u32);
                                for cell in &cells[..len] {
                                    let _ = window[written].write(*cell);
                                    written += 1;
                                }
                            }
                            Err(failure) => {
                                error.record(row, failure);
                                return;
                            }
                        }
                    }
                    debug_assert_eq!(written, window.len());
                });
            if let Some(failure) = error.take() {
                return Err(failure.into());
            }
        }
        // SAFETY: exclusive windows cover all `total` initialized slots.
        unsafe { entries.set_len(total) };
        Ok(CollectRegisterEntries {
            entries,
            rs1_indices,
            rs2_indices,
            rd_indices,
            rd_inc,
        })
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
        CapturedState, JoltInstructionKind, JoltInstructionRow, JoltTraceRow, NonMemoryState,
        NormalizedOperands,
    };

    use super::*;

    #[test]
    fn rejects_register_outside_protocol_domain() {
        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            operands: NormalizedOperands {
                rs1: Some(200),
                ..Default::default()
            },
            ..Default::default()
        };
        let row = JoltTraceRow::from_components(
            CapturedState::NonMemory(NonMemoryState::default()),
            &instruction,
            0,
        )
        .unwrap();
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::default(),
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
