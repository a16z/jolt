use core::mem::MaybeUninit;
use std::borrow::Cow;

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

use super::sparse::{SeedEntry, MAX_ENTRIES_PER_CYCLE};
use crate::optimized::support::BundleAccess;
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
pub(crate) struct SharedRdIndices {
    pub(crate) indices: Vec<Option<u8>>,
    pub(crate) cycles: usize,
}

impl SharedRdIndices {
    pub(crate) fn new(mut indices: Vec<Option<u8>>, cycles: usize) -> Self {
        assert!(indices.len() <= cycles);
        let prefix = indices
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |i| i + 1);
        indices.truncate(prefix);
        indices.shrink_to_fit();
        Self { indices, cycles }
    }
}

const COLLECT_CHUNK: usize = 1 << 16;

/// Cycles per chunk of the parallel collect.
const COLLECT_PAR_CHUNK: usize = 1 << 14;

/// Signed rd write delta, or zero without an rd operand.
#[inline]
pub(super) fn raw_rd_inc(cycle: &RegisterCycleRow) -> i128 {
    match cycle.rd {
        Some((_, pre, post)) => post as i128 - pre as i128,
        None => 0,
    }
}

pub(super) struct CollectRegisterEntries<E = Vec<SeedEntry>> {
    pub(super) entries: E,
    pub(super) rs1_indices: Vec<Option<u8>>,
    pub(super) rs2_indices: Vec<Option<u8>>,
    pub(super) rd_indices: Vec<Option<u8>>,
    pub(super) rd_inc: Vec<i128>,
}

/// The canonical collector's entry writer. Allocation belongs to the consumer;
/// count, extraction and scatter remain identical for CPU and Metal storage.
pub(super) enum SeedEntryProducer<'a> {
    Stored(Cow<'a, [SeedEntry]>),
    #[cfg(feature = "parallel")]
    Trace {
        access: &'a RandomAccessRows,
        cycles: usize,
        counts: &'a [usize],
        len: usize,
    },
}

impl SeedEntryProducer<'_> {
    #[cfg(any(feature = "parallel", feature = "metal"))]
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Stored(entries) => entries.len(),
            #[cfg(feature = "parallel")]
            Self::Trace { len, .. } => *len,
        }
    }

    /// On success every slot is initialized, including the seed's explicit
    /// reserved byte. An error leaves an unspecified subset initialized.
    #[cfg(any(feature = "parallel", feature = "metal"))]
    pub(super) fn write<F: JoltField>(
        &self,
        destination: &mut [MaybeUninit<SeedEntry>],
    ) -> Result<(), KernelError<F>> {
        if destination.len() != self.len() {
            return Err(KernelError::InvariantViolation {
                reason: "register seed destination has the wrong length",
            });
        }
        match self {
            Self::Stored(entries) => {
                for (slot, entry) in destination.iter_mut().zip(entries.iter()) {
                    let _ = slot.write(*entry);
                }
            }
            #[cfg(feature = "parallel")]
            Self::Trace {
                access,
                cycles,
                counts,
                ..
            } => {
                let mut rest = destination;
                let mut windows = Vec::with_capacity(counts.len());
                for &count in *counts {
                    let (head, tail) = rest.split_at_mut(count);
                    windows.push(head);
                    rest = tail;
                }
                if !rest.is_empty() {
                    return Err(KernelError::InvariantViolation {
                        reason: "register seed counts do not cover the destination",
                    });
                }
                let error = FirstErrorLatch::new();
                windows
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(chunk_index, window)| {
                        let base = chunk_index * COLLECT_PAR_CHUNK;
                        let top = ((chunk_index + 1) * COLLECT_PAR_CHUNK).min(*cycles);
                        let mut written = 0usize;
                        for row in base..top {
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
                        if written != window.len() {
                            error.record(
                                base,
                                WitnessError::InvalidWitnessData {
                                    label: "register seed collection",
                                    reason: "entry count changed between count and scatter".into(),
                                },
                            );
                        }
                    });
                if let Some(failure) = error.take() {
                    return Err(failure.into());
                }
            }
        }
        Ok(())
    }

    pub(super) fn into_vec<F: JoltField>(self) -> Result<Vec<SeedEntry>, KernelError<F>> {
        match self {
            Self::Stored(entries) => Ok(entries.into_owned()),
            #[cfg(feature = "parallel")]
            producer @ Self::Trace { .. } => {
                let len = producer.len();
                let mut entries = Vec::with_capacity(len);
                producer.write(&mut entries.spare_capacity_mut()[..len])?;
                // SAFETY: write succeeded and initializes the entire exact-length span.
                unsafe { entries.set_len(len) };
                Ok(entries)
            }
        }
    }
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
    /// Bytes of the parallel collect's bookkeeping over `cycles` cycles, live
    /// while the entries scatter, or `None` past `usize::MAX`: a count and an
    /// output window per chunk.
    pub(super) fn chunk_bytes(cycles: usize) -> Option<usize> {
        cycles
            .div_ceil(COLLECT_PAR_CHUNK)
            .checked_mul(size_of::<usize>() + size_of::<&mut [MaybeUninit<SeedEntry>]>())
    }

    pub(super) fn collect<F: JoltField, E>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
        produce: impl FnOnce(SeedEntryProducer<'_>) -> Result<E, KernelError<F>>,
    ) -> Result<CollectRegisterEntries<E>, KernelError<F>> {
        if u32::try_from(cycles.saturating_sub(1)).is_err() {
            return Err(KernelError::InvariantViolation {
                reason: "cycle count exceeds the seed entries' packed u32 row domain",
            });
        }
        let mut retained = cycles;
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                let prefix = BundleAccess::<RegisterCycleRow>::View(&access)
                    .constant_suffix_start(cycles)?;
                if prefix < cycles
                    && access.window::<RegisterCycleRow>(prefix)? == RegisterCycleRow::default()
                {
                    retained = prefix;
                }
            }
            #[cfg(feature = "parallel")]
            if cycles <= access.cycles() {
                return Self::collect_par(&access, retained, produce);
            }
        }
        let cycles = retained;
        let mut consumers = (CollectRegisterEntries {
            entries: Vec::with_capacity(cycles * MAX_ENTRIES_PER_CYCLE),
            rs1_indices: Vec::with_capacity(cycles),
            rs2_indices: Vec::with_capacity(cycles),
            rd_indices: Vec::with_capacity(cycles),
            rd_inc: Vec::with_capacity(cycles),
        },);
        stream_witnesses(witness, 0..cycles, COLLECT_CHUNK, &mut consumers)?;
        let Self {
            entries,
            rs1_indices,
            rs2_indices,
            rd_indices,
            rd_inc,
        } = consumers.0;
        Ok(CollectRegisterEntries {
            entries: produce(SeedEntryProducer::Stored(Cow::Owned(entries)))?,
            rs1_indices,
            rs2_indices,
            rd_indices,
            rd_inc,
        })
    }

    /// Two-pass parallel build: count and fill columns, then scatter entries.
    /// Exclusive offsets avoid a second entry-sized allocation.
    #[cfg(feature = "parallel")]
    fn collect_par<F: JoltField, E>(
        access: &RandomAccessRows,
        cycles: usize,
        produce: impl FnOnce(SeedEntryProducer<'_>) -> Result<E, KernelError<F>>,
    ) -> Result<CollectRegisterEntries<E>, KernelError<F>> {
        let mut rs1_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rs2_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rd_indices: Vec<Option<u8>> = Vec::with_capacity(cycles);
        let mut rd_inc: Vec<i128> = Vec::with_capacity(cycles);
        let error = FirstErrorLatch::new();
        let mut counts: Vec<usize> = Vec::new();
        (
            rs1_indices.spare_capacity_mut()[..cycles].par_chunks_mut(COLLECT_PAR_CHUNK),
            rs2_indices.spare_capacity_mut()[..cycles].par_chunks_mut(COLLECT_PAR_CHUNK),
            rd_indices.spare_capacity_mut()[..cycles].par_chunks_mut(COLLECT_PAR_CHUNK),
            rd_inc.spare_capacity_mut()[..cycles].par_chunks_mut(COLLECT_PAR_CHUNK),
        )
            .into_par_iter()
            .enumerate()
            .map(|(chunk_index, (rs1, rs2, rd, inc))| {
                let base = chunk_index * COLLECT_PAR_CHUNK;
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

        let entries = produce(SeedEntryProducer::Trace {
            access,
            cycles,
            len: counts.iter().sum(),
            counts: &counts,
        })?;
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
