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
}

/// Setup requests of a limited trace, admitted together before any of them
/// is made.
pub(super) struct LimitedSetup {
    limit: usize,
    rows: usize,
    window: usize,
    output_bytes: usize,
    /// The caller's input and advice buffers, which outlive the trace.
    caller_bytes: usize,
}

impl LimitedSetup {
    /// Checks the device's input copies and output reserve, the bytecode
    /// preprocessing, flat guest memory, the ELF copy setup makes, the initial
    /// row reserve and the pending window beside the caller's buffers and
    /// advice tape, and the preprocessing's peak with its bytecode copy beside
    /// those buffers alone.
    pub(super) fn admit(
        limit: TraceLimit,
        program: &JoltProgram,
        inputs: &TraceInputs,
        batch_rows: usize,
    ) -> Result<Self, TraceError> {
        let phase = TraceAllocationPhase::Initial;
        let overflow = TraceAllocationError::SizeOverflow { phase };
        let layout = MemoryLayout::try_new(&inputs.memory_config)?;
        let output_bytes = usize::try_from(layout.max_output_size).map_err(|_| overflow)?;
        let memory_words = flat_doublewords(Emulator::memory_capacity(
            program.elf_bytes(),
            Some(&layout),
        ));
        let rows = limit.initial_rows;
        let window = pending_window(batch_rows).ok_or(overflow)?;
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
                BytecodePreprocessing::retained_bytes(&program.expanded_bytecode)?,
                bytes_of::<u64>(usize::try_from(memory_words).map_err(|_| overflow)?, phase)?,
                // `ElfAnalyzer` copies the ELF for the length of setup.
                program.elf_bytes().len(),
                bytes_of::<JoltTraceRow>(rows, phase)?,
                bytes_of::<Cycle>(window, phase)?,
            ],
            phase,
        )?;
        // `preprocess` frees its copy of the bytecode before any other setup
        // request, so the copy needs room only beside the caller's buffers.
        let preprocess =
            BytecodePreprocessing::required_allocation_bytes(&program.expanded_bytecode)?;
        let held = caller_bytes.saturating_add(tape_bytes);
        TraceAllocationAllowance::new(limit.bytes, 0).check_request(
            phase,
            held,
            requests.max(preprocess),
        )?;
        Ok(Self {
            limit: limit.bytes,
            rows,
            window,
            output_bytes,
            caller_bytes,
        })
    }

    /// Makes the admitted output, row and pending-window reservations and
    /// counts the setup buffers at their observed capacity.
    #[expect(clippy::expect_used, reason = "create_emulator installs the device")]
    pub(super) fn reserve(
        self,
        emulator: &mut Emulator,
        bytecode: &BytecodePreprocessing,
    ) -> Result<LimitedTrace, TraceAllocationError> {
        let phase = TraceAllocationPhase::Initial;
        let cpu = emulator.get_mut_cpu();
        let device = cpu
            .mmu
            .jolt_device
            .as_mut()
            .expect("JoltDevice was not initialized");
        reserve_exact(&mut device.outputs, self.output_bytes, phase)?;
        let mut rows = Vec::new();
        reserve_exact(&mut rows, self.rows, phase)?;
        let mut cycles = Vec::new();
        reserve_exact(&mut cycles, self.window, phase)?;
        let resident = [
            bytecode.heap_bytes(),
            self.caller_bytes,
            heap_bytes(&device.inputs),
            heap_bytes(&device.untrusted_advice),
            heap_bytes(&device.trusted_advice),
            heap_bytes(&device.outputs),
            cpu.mmu.memory.memory.flat_heap_bytes(),
        ]
        .into_iter()
        .fold(0, usize::saturating_add);
        Ok(LimitedTrace {
            limit: self.limit,
            resident,
            cycles,
            rows,
            chunks: Vec::new(),
            field_bytes: 0,
        })
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
