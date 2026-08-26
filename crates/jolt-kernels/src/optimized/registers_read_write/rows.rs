//! Typed per-cycle register rows and sparse-entry collection.

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

use super::sparse::IndexedSparseEntry;
use crate::KernelError;

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
    // `TraceRow` is nameable from this crate only through the doc-hidden
    // re-export the bundle derive uses; jolt-kernels deliberately has no
    // jolt-program dependency.
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

/// Cross-member carry: the per-cycle `rd` hot indices, parked by this kernel's
/// `prepare` for the stage-5 val-evaluation kernel (which otherwise re-walks
/// the trace to collect them).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedRdIndices(pub Vec<Option<u8>>);

/// The row-window size of the streaming entry-collection pass (matches
/// `support::collect_rows`: wide enough to amortize the per-chunk rayon
/// extraction dispatch).
const COLLECT_CHUNK: usize = 1 << 16;

/// Streaming consumer building the sparse entries and the operand index
/// columns in one trace pass, no whole-trace row materialization.
pub(super) struct CollectRegisterEntries<F: JoltField> {
    pub(super) entries: Vec<IndexedSparseEntry<F>>,
    pub(super) rs1_indices: Vec<Option<u8>>,
    pub(super) rs2_indices: Vec<Option<u8>>,
    pub(super) rd_indices: Vec<Option<u8>>,
}

impl<F: JoltField> StreamConsumer for CollectRegisterEntries<F> {
    type Witness = RegisterCycleRow;

    fn consume(&mut self, chunk: &[RegisterCycleRow]) {
        for cycle in chunk {
            let row = self.rs1_indices.len();
            debug_assert!(u32::try_from(row).is_ok());
            let (cells, len) = cycle.entries(row as u32);
            self.entries.extend_from_slice(&cells[..len]);
            self.rs1_indices.push(cycle.rs1.map(|(k, _)| k));
            self.rs2_indices.push(cycle.rs2.map(|(k, _)| k));
            self.rd_indices.push(cycle.rd.map(|(k, ..)| k));
        }
    }
}

impl<F: JoltField> CollectRegisterEntries<F> {
    /// Builds the sparse entries and the operand index columns in one trace
    /// pass. Slice-backed sources build index-parallel; re-emulating sources
    /// stream sequentially. Entry values and order are identical either way —
    /// [`RegisterCycleRow::entries`] is pure per cycle, and runs concatenate in
    /// cycle order.
    pub(super) fn collect(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        #[cfg(feature = "parallel")]
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                return Self::collect_par(&access, cycles);
            }
        }
        let mut consumers = (Self {
            entries: Vec::with_capacity(cycles * 3),
            rs1_indices: Vec::with_capacity(cycles),
            rs2_indices: Vec::with_capacity(cycles),
            rd_indices: Vec::with_capacity(cycles),
        },);
        stream_witnesses(witness, 0..cycles, COLLECT_CHUNK, &mut consumers)?;
        Ok(consumers.0)
    }

    /// The index-parallel entry build: a first pass counts each chunk's
    /// entries (extraction-only, no staging), so entries scatter straight into
    /// their exclusive-scan offsets on the second pass — no per-chunk runs, no
    /// co-resident copy (the entry vector is the stage's largest allocation;
    /// briefly doubling it moves the prover's peak). The three operand index
    /// columns fill on the counting pass. Entry values and order are identical
    /// to the streaming pass: [`RegisterCycleRow::entries`] is pure per cycle.
    #[cfg(feature = "parallel")]
    fn collect_par(access: &RandomAccessRows, cycles: usize) -> Result<Self, KernelError<F>> {
        use core::mem::MaybeUninit;
        /// The scatter grain (matches the whole-range collectors' load-balance
        /// tradeoff at ~3 entries per cycle).
        const CHUNK: usize = 1 << 14;
        let mut rs1_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rs2_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rd_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let error = FirstErrorLatch::new();
        let chunk_count = cycles.div_ceil(CHUNK);
        // Pass 1: count entries per chunk and fill the index columns.
        let mut counts: Vec<usize> = Vec::new();
        (
            rs1_indices.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            rs2_indices.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            rd_indices.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
        )
            .into_par_iter()
            .enumerate()
            .map(|(chunk_index, (rs1, rs2, rd))| {
                let base = chunk_index * CHUNK;
                let mut count = 0usize;
                for offset in 0..rs1.len() {
                    match access.window::<RegisterCycleRow>(base + offset) {
                        Ok(cycle) => {
                            count += cycle.entry_count();
                            let _ = rs1[offset].write(cycle.rs1.map(|(k, _)| k));
                            let _ = rs2[offset].write(cycle.rs2.map(|(k, _)| k));
                            let _ = rd[offset].write(cycle.rd.map(|(k, ..)| k));
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
        // SAFETY: the error latch is empty, so every chunk ran to completion and
        // initialized its whole span of all three index columns.
        unsafe {
            rs1_indices.set_len(cycles);
            rs2_indices.set_len(cycles);
            rd_indices.set_len(cycles);
        }

        // Exclusive scan of the per-chunk counts, then pass 2: re-extract and
        // scatter entries straight to their offsets.
        let mut offsets: Vec<usize> = Vec::with_capacity(chunk_count);
        let mut total = 0usize;
        for &count in &counts {
            offsets.push(total);
            total += count;
        }
        let mut entries: Vec<IndexedSparseEntry<F>> = Vec::with_capacity(total);
        {
            let mut rest: &mut [MaybeUninit<IndexedSparseEntry<F>>] =
                &mut entries.spare_capacity_mut()[..total];
            let mut windows: Vec<&mut [MaybeUninit<IndexedSparseEntry<F>>]> =
                Vec::with_capacity(chunk_count);
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
                        // Pass 1 latched any extraction error; a second window
                        // over the same immutable rows cannot fail differently,
                        // but stay conservative and latch again.
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
        // SAFETY: the windows partition exactly `total` slots (the exclusive
        // scan of the same counts pass 2 reproduces), and every window was
        // fully written above.
        unsafe { entries.set_len(total) };
        Ok(Self {
            entries,
            rs1_indices,
            rs2_indices,
            rd_indices,
        })
    }
}

#[cfg(feature = "parallel")]
impl RegisterCycleRow {
    /// The entry count [`Self::entries`] will produce for one cycle — the
    /// counting pass's cheap twin (kept adjacent so the merge rules stay in
    /// sync: rs2 folds into rs1's entry, rd into either read's).
    fn entry_count(&self) -> usize {
        let mut count = 0usize;
        let mut cols = [None; 2];
        if let Some((rs1, _)) = self.rs1 {
            cols[0] = Some(rs1);
            count += 1;
        }
        if let Some((rs2, _)) = self.rs2 {
            if cols[0] != Some(rs2) {
                cols[1] = Some(rs2);
                count += 1;
            }
        }
        if let Some((rd, ..)) = self.rd {
            if cols[0] != Some(rd) && cols[1] != Some(rd) {
                count += 1;
            }
        }
        count
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

    /// The counting pass and the write pass must agree on every operand
    /// pattern — including raw index 255, which the old `u8::MAX` sentinel
    /// collided with (count 0, write 1 → pass-2 window overrun).
    #[cfg(feature = "parallel")]
    #[test]
    fn cycle_entry_count_matches_cycle_entries() {
        use jolt_field::Fr;
        let candidates: [Option<u8>; 5] = [None, Some(0), Some(5), Some(127), Some(255)];
        for rs1 in candidates {
            for rs2 in candidates {
                for rd in candidates {
                    let cycle = RegisterCycleRow {
                        rs1: rs1.map(|register| (register, 11)),
                        rs2: rs2.map(|register| (register, 22)),
                        rd: rd.map(|register| (register, 33, 44)),
                    };
                    let (_, len) = cycle.entries::<Fr>(0);
                    assert_eq!(
                        cycle.entry_count(),
                        len,
                        "count/write divergence for {cycle:?}"
                    );
                }
            }
        }
    }

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
