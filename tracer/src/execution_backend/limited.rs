#[cfg(feature = "field-inline")]
use super::field_event_count;
use std::mem::MaybeUninit;

use super::{
    finish_rows, heap_bytes, pending_window, ConvertedChunk, PARALLEL_ROW_CONVERSION_THRESHOLD,
};
use crate::emulator::memory::{flat_doublewords, Memory};
use crate::emulator::Emulator;
use crate::instruction::Cycle;
use crate::trace_allocation::{
    bytes_of, growth_target, reserve_exact, sum_bytes, TraceAllocationAllowance,
};
use common::jolt_device::MemoryLayout;
#[cfg(feature = "field-inline")]
use jolt_program::execution::FieldEvent;
use jolt_program::execution::{
    ExecutionDimensions, JoltProgram, MemoryImage, TraceAllocationError, TraceAllocationPhase,
    TraceData, TraceError, TraceInputs,
};
use jolt_program::preprocess::BytecodePreprocessing;
use jolt_riscv::JoltTraceRow;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct TraceLimit {
    pub bytes: usize,
    pub initial_rows: usize,
    pub batch_rows: usize,
}

impl TraceLimit {
    pub(super) fn admit(
        self,
        program: &JoltProgram,
        inputs: &TraceInputs,
    ) -> Result<LimitedSetup, TraceError> {
        let phase = TraceAllocationPhase::Initial;
        let window =
            pending_window(self.batch_rows).ok_or(TraceAllocationError::SizeOverflow { phase })?;
        let storage = sum_bytes(
            [
                BytecodePreprocessing::retained_bytes(&program.expanded_bytecode)?,
                bytes_of::<JoltTraceRow>(self.initial_rows, phase)?,
                bytes_of::<Cycle>(window, phase)?,
            ],
            phase,
        )?;
        let preprocess =
            BytecodePreprocessing::required_allocation_bytes(&program.expanded_bytecode)?;
        LimitedSetup::admit(self.bytes, program, inputs, storage, preprocess)
    }
}

/// Setup requests of a limited run, admitted together with the run's own
/// setup storage before any of them is made.
pub(super) struct LimitedSetup {
    output_bytes: usize,
    /// The caller's input and advice buffers, which outlive the run.
    caller_bytes: usize,
}

impl LimitedSetup {
    /// Checks the device's input copies and output reserve, flat guest
    /// memory, the ELF copy setup makes and `storage` bytes of the run's own
    /// setup requests beside the caller's buffers and advice tape, and the
    /// `preprocess` bytes its bytecode preprocessing requests at most, before
    /// any of those, beside the caller's buffers and tape alone.
    pub(super) fn admit(
        limit: usize,
        program: &JoltProgram,
        inputs: &TraceInputs,
        storage: usize,
        preprocess: usize,
    ) -> Result<Self, TraceError> {
        let phase = TraceAllocationPhase::Initial;
        let overflow = TraceAllocationError::SizeOverflow { phase };
        let layout = MemoryLayout::try_new(&inputs.memory_config)?;
        let output_bytes = usize::try_from(layout.max_output_size).map_err(|_| overflow)?;
        let memory_words = flat_doublewords(Emulator::memory_capacity(
            program.elf_bytes(),
            Some(&layout),
        ));
        let caller_bytes = [
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
        ]
        .into_iter()
        .map(heap_bytes)
        .fold(0, usize::saturating_add);
        let tape_bytes = inputs.advice_tape.as_ref().map_or(0, heap_bytes);
        let requests = sum_bytes(
            [
                inputs.inputs.len(),
                inputs.untrusted_advice.len(),
                inputs.trusted_advice.len(),
                output_bytes,
                bytes_of::<u64>(usize::try_from(memory_words).map_err(|_| overflow)?, phase)?,
                // `ElfAnalyzer` copies the ELF for the length of setup.
                program.elf_bytes().len(),
                storage,
            ],
            phase,
        )?;
        // The bytecode preprocessing frees its copy of the bytecode before any
        // other setup request, so its peak needs room only beside the caller's
        // buffers.
        let held = caller_bytes.saturating_add(tape_bytes);
        TraceAllocationAllowance::new(limit, 0).check_request(
            phase,
            held,
            requests.max(preprocess),
        )?;
        Ok(Self {
            output_bytes,
            caller_bytes,
        })
    }

    /// Makes the admitted output reservation and returns the bytes the
    /// caller's buffers, the device's input copies and outputs and flat guest
    /// memory hold at their observed capacity.
    #[expect(clippy::expect_used, reason = "create_emulator installs the device")]
    pub(super) fn reserve_device(
        &self,
        emulator: &mut Emulator,
    ) -> Result<usize, TraceAllocationError> {
        let phase = TraceAllocationPhase::Initial;
        let cpu = emulator.get_mut_cpu();
        let device = cpu
            .mmu
            .jolt_device
            .as_mut()
            .expect("JoltDevice was not initialized");
        reserve_exact(&mut device.outputs, self.output_bytes, phase)?;
        Ok([
            self.caller_bytes,
            heap_bytes(&device.inputs),
            heap_bytes(&device.untrusted_advice),
            heap_bytes(&device.trusted_advice),
            heap_bytes(&device.outputs),
            cpu.mmu.memory.memory.flat_heap_bytes(),
        ]
        .into_iter()
        .fold(0, usize::saturating_add))
    }
}

/// A limited trace's storage, every buffer counted at its observed capacity.
pub(super) struct LimitedTrace {
    pub(super) limit: usize,
    /// The bytecode preprocessing, the caller's buffers, the device's input
    /// copies and output reserve, and flat guest memory, none of which grows
    /// after setup.
    pub(super) resident: usize,
    /// The pending cycles, in a window that holds one full tick past the
    /// batch.
    pub(super) cycles: Vec<Cycle>,
    pub(super) rows: Vec<JoltTraceRow>,
    pub(super) chunks: Vec<ConvertedChunk>,
    /// Total capacity bytes of the chunks' field-event buffers, updated only
    /// after a successful conversion; summing all chunks per batch is quadratic.
    pub(super) field_bytes: usize,
}

impl LimitedTrace {
    /// Makes the admitted reservations of `rows` rows and a `window`-cycle
    /// pending window beside `resident` bytes.
    pub(super) fn reserve(
        limit: TraceLimit,
        resident: usize,
    ) -> Result<Self, TraceAllocationError> {
        let phase = TraceAllocationPhase::Initial;
        let rows = limit.initial_rows;
        let window =
            pending_window(limit.batch_rows).ok_or(TraceAllocationError::SizeOverflow { phase })?;
        let mut row_buffer = Vec::new();
        reserve_exact(&mut row_buffer, rows, phase)?;
        let mut cycles = Vec::new();
        reserve_exact(&mut cycles, window, phase)?;
        Ok(Self {
            limit: limit.bytes,
            resident,
            cycles,
            rows: row_buffer,
            chunks: Vec::new(),
            field_bytes: 0,
        })
    }

    /// What an advice write may request beside every buffer of this trace;
    /// the write's own check counts the advice tape.
    pub(super) fn advice_write_allowance(&self) -> TraceAllocationAllowance {
        let live = [
            self.resident,
            heap_bytes(&self.cycles),
            heap_bytes(&self.rows),
            heap_bytes(&self.chunks),
            self.field_bytes,
        ]
        .into_iter()
        .fold(0, usize::saturating_add);
        TraceAllocationAllowance::new(self.limit, live)
    }

    /// What a request may take beside the resident buffers, the pending
    /// window, the chunks' field events and `tape` bytes of advice tape.
    fn others(&self, tape: usize) -> TraceAllocationAllowance {
        let live = [
            self.resident,
            heap_bytes(&self.cycles),
            self.field_bytes,
            tape,
        ]
        .into_iter()
        .fold(0, usize::saturating_add);
        TraceAllocationAllowance::new(self.limit, live)
    }

    /// Appends the rows of the pending cycles, which follow every converted
    /// row, and one chunk per conversion slice, then clears the cycles. Row
    /// growth is checked with the old and new row buffers and the chunk
    /// metadata live; one request then covers the metadata growth and every
    /// slice's field events, which the slices' workers reserve exactly.
    pub(super) fn convert_pending(
        &mut self,
        bytecode: &BytecodePreprocessing,
        tape: usize,
    ) -> Result<(), TraceError> {
        let others = self.others(tape);
        let cycles = &self.cycles;
        let parallel = cycles.len() > PARALLEL_ROW_CONVERSION_THRESHOLD;
        let _span = tracing::info_span!(
            "trace_rows_from_cycles",
            cycles = cycles.len(),
            mode = if parallel { "parallel" } else { "serial" }
        )
        .entered();
        let slices = if parallel {
            cycles.len().div_ceil(PARALLEL_ROW_CONVERSION_THRESHOLD)
        } else {
            1
        };
        let phase = TraceAllocationPhase::RowGrowth;
        let required = self
            .rows
            .len()
            .checked_add(cycles.len())
            .ok_or(TraceAllocationError::SizeOverflow { phase })?;
        if required > self.rows.capacity() {
            let target = growth_target(self.rows.capacity(), required);
            let held = heap_bytes(&self.rows).saturating_add(heap_bytes(&self.chunks));
            others.check_request(phase, held, bytes_of::<JoltTraceRow>(target, phase)?)?;
            let additional = target - self.rows.len();
            reserve_exact(&mut self.rows, additional, phase)?;
        }
        let phase = TraceAllocationPhase::Conversion;
        let overflow = TraceAllocationError::SizeOverflow { phase };
        let required = self.chunks.len().checked_add(slices).ok_or(overflow)?;
        let target = (required > self.chunks.capacity())
            .then(|| growth_target(self.chunks.capacity(), required));
        let new_chunks = bytes_of::<ConvertedChunk>(target.unwrap_or(0), phase)?;
        #[cfg(feature = "field-inline")]
        let new_fields = bytes_of::<FieldEvent>(field_event_count(cycles), phase)?;
        #[cfg(not(feature = "field-inline"))]
        let new_fields = 0;
        let held = heap_bytes(&self.rows).saturating_add(heap_bytes(&self.chunks));
        others.check_request(
            phase,
            held,
            new_chunks.checked_add(new_fields).ok_or(overflow)?,
        )?;
        if let Some(target) = target {
            let additional = target - self.chunks.len();
            reserve_exact(&mut self.chunks, additional, phase)?;
        }
        let start = self.rows.len();
        let first_chunk = self.chunks.len();
        self.chunks
            .resize_with(first_chunk + slices, ConvertedChunk::default);
        let chunks = &mut self.chunks[first_chunk..];
        let slots = &mut self.rows.spare_capacity_mut()[..cycles.len()];
        let fill = |first: usize,
                    chunk: &mut ConvertedChunk,
                    slots: &mut [MaybeUninit<JoltTraceRow>],
                    cycles: &[Cycle]| {
            #[cfg(feature = "field-inline")]
            reserve_exact(&mut chunk.fields, field_event_count(cycles), phase)?;
            chunk.fill(first, slots, cycles, bytecode)
        };
        if parallel {
            chunks
                .par_iter_mut()
                .zip(slots.par_chunks_mut(PARALLEL_ROW_CONVERSION_THRESHOLD))
                .zip(cycles.par_chunks(PARALLEL_ROW_CONVERSION_THRESHOLD))
                .enumerate()
                .try_for_each(|(index, ((chunk, slots), cycles))| {
                    fill(
                        start + index * PARALLEL_ROW_CONVERSION_THRESHOLD,
                        chunk,
                        slots,
                        cycles,
                    )
                })?;
        } else {
            fill(start, &mut chunks[0], slots, cycles)?;
        }
        // SAFETY: every worker returned, and all of them initialized their
        // slots: the `cycles.len()` slots from `start`. JoltTraceRow is Copy,
        // so partially written rows need no cleanup on error or unwinding.
        unsafe { self.rows.set_len(start + cycles.len()) };
        #[cfg(feature = "field-inline")]
        {
            let new_bytes = self.chunks[first_chunk..]
                .iter()
                .map(|chunk| heap_bytes(&chunk.fields))
                .fold(0, usize::saturating_add);
            self.field_bytes = self.field_bytes.saturating_add(new_bytes);
        }
        self.cycles.clear();
        Ok(())
    }

    /// The trace data, its dimensions and the final memory image. The merged
    /// field events are checked with every source buffer, the pending window
    /// and `tape` bytes of advice tape live; the image after the window drops,
    /// beside the trace data, the tape and the resident buffers.
    pub(super) fn finish(
        self,
        final_memory: &Memory,
        tape: usize,
    ) -> Result<(TraceData, ExecutionDimensions, MemoryImage), TraceError> {
        #[cfg(feature = "field-inline")]
        let fields = {
            let phase = TraceAllocationPhase::FieldMerge;
            let count = self.chunks.iter().map(|chunk| chunk.fields.len()).sum();
            let held = heap_bytes(&self.rows).saturating_add(heap_bytes(&self.chunks));
            self.others(tape)
                .check_request(phase, held, bytes_of::<FieldEvent>(count, phase)?)?;
            let mut fields = Vec::new();
            reserve_exact(&mut fields, count, phase)?;
            fields
        };
        let Self {
            limit,
            resident,
            cycles,
            rows,
            chunks,
            ..
        } = self;
        let (data, dimensions) = finish_rows(
            rows,
            chunks,
            #[cfg(feature = "field-inline")]
            fields,
        )?;
        // Freed before the image request, which does not count it.
        drop(cycles);
        let live = [resident, tape, data.heap_bytes()]
            .into_iter()
            .fold(0, usize::saturating_add);
        let bytes =
            final_memory.try_flat_nonzero_bytes(TraceAllocationAllowance::new(limit, live))?;
        Ok((data, dimensions, MemoryImage { bytes }))
    }
}
