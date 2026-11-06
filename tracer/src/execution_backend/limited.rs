#[cfg(feature = "field-inline")]
use super::field_event_count;
use std::mem::MaybeUninit;

use super::{
    conversion_slices, finish_rows, heap_bytes, next_emission, pending_window, ConvertedChunk,
    FIELD_EVENT_BYTES, PARALLEL_ROW_CONVERSION_THRESHOLD, TICK_BOUND,
};
use crate::emulator::cpu::Cpu;
use crate::emulator::memory::FlatImage;
use crate::emulator::memory::{flat_doublewords, Memory};
use crate::emulator::Emulator;
use crate::instruction::{Cycle, Emission};
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
use jolt_program::expand::MAX_EXPANDED_ROWS_PER_SOURCE;
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
    /// copies and output reserve, and flat guest memory (or the final image
    /// collected in its place), none of which grows after setup.
    pub(super) resident: usize,
    /// The pending cycles: every cycle while the trace accumulates, otherwise
    /// a window that holds one full tick past the batch.
    pub(super) cycles: Vec<Cycle>,
    pub(super) rows: Vec<JoltTraceRow>,
    pub(super) chunks: Vec<ConvertedChunk>,
    /// Total capacity bytes of the chunks' field-event buffers, updated only
    /// after a successful conversion; summing all chunks per batch is quadratic.
    pub(super) field_bytes: usize,
}

/// A limited trace's buffers as its request checks count them, in bytes at
/// observed capacity, or at the capacity a planned request gives them.
#[derive(Clone, Copy)]
struct Residency {
    limit: usize,
    resident: usize,
    tape: usize,
    cycles: usize,
    rows: usize,
    /// Converted chunks, and the metadata's capacity in chunks.
    chunks: usize,
    chunk_capacity: usize,
    field_bytes: usize,
}

impl Residency {
    /// What a request may take beside the resident buffers, the cycles, the
    /// chunks' field events and the advice tape.
    fn others(&self) -> TraceAllocationAllowance {
        let live = [self.resident, self.cycles, self.field_bytes, self.tape]
            .into_iter()
            .fold(0, usize::saturating_add);
        TraceAllocationAllowance::new(self.limit, live)
    }

    /// The rows and the chunk metadata.
    fn held(&self) -> usize {
        self.chunk_capacity
            .saturating_mul(size_of::<ConvertedChunk>())
            .saturating_add(self.rows)
    }

    fn live(&self) -> usize {
        let others = [self.resident, self.cycles, self.field_bytes, self.tape];
        others.into_iter().fold(self.held(), usize::saturating_add)
    }

    /// Checks a `capacity`-cycle buffer beside every buffer.
    fn grow_cycles(&self, capacity: usize) -> Result<usize, TraceAllocationError> {
        let phase = TraceAllocationPhase::CycleBuffer;
        let cycles = bytes_of::<Cycle>(capacity, phase)?;
        self.others().check_request(phase, self.held(), cycles)
    }

    /// Checks `capacity` rows beside every buffer. Returns the bytes live then
    /// and the residency once they replace the old rows.
    fn grow_rows(self, capacity: usize) -> Result<(usize, Self), TraceAllocationError> {
        let phase = TraceAllocationPhase::RowGrowth;
        let rows = bytes_of::<JoltTraceRow>(capacity, phase)?;
        let live = self.others().check_request(phase, self.held(), rows)?;
        Ok((live, Self { rows, ..self }))
    }

    /// The chunks after `slices` more, and the metadata capacity they grow it
    /// to by [`growth_target`], if they must.
    fn chunk_growth(&self, slices: usize) -> Result<(usize, Option<usize>), TraceAllocationError> {
        let phase = TraceAllocationPhase::Conversion;
        let chunks = self
            .chunks
            .checked_add(slices)
            .ok_or(TraceAllocationError::SizeOverflow { phase })?;
        let target =
            (chunks > self.chunk_capacity).then(|| growth_target(self.chunk_capacity, chunks));
        Ok((chunks, target))
    }

    /// Checks one conversion's request beside every buffer: the metadata of
    /// `slices` more chunks, grown with the old live, and `events` field
    /// events. Returns the bytes live then and the residency after it.
    fn convert(self, slices: usize, events: usize) -> Result<(usize, Self), TraceAllocationError> {
        let phase = TraceAllocationPhase::Conversion;
        let overflow = TraceAllocationError::SizeOverflow { phase };
        let (chunks, target) = self.chunk_growth(slices)?;
        let metadata = bytes_of::<ConvertedChunk>(target.unwrap_or(0), phase)?;
        let fields = events.checked_mul(FIELD_EVENT_BYTES).ok_or(overflow)?;
        let request = metadata.checked_add(fields).ok_or(overflow)?;
        let live = self.others().check_request(phase, self.held(), request)?;
        let converted = Self {
            chunks,
            chunk_capacity: target.unwrap_or(self.chunk_capacity),
            field_bytes: self.field_bytes.saturating_add(fields),
            ..self
        };
        Ok((live, converted))
    }

    /// Checks converting `pending` cycles with `events` field events, the rows
    /// grown first to `rows` when given. Returns the most bytes live during
    /// either request and the residency after them.
    fn plan_conversion(
        self,
        pending: usize,
        events: usize,
        rows: Option<usize>,
    ) -> Result<(usize, Self), TraceAllocationError> {
        let (growth, grown) = match rows {
            Some(capacity) => self.grow_rows(capacity)?,
            None => (0, self),
        };
        let (conversion, converted) = grown.convert(conversion_slices(pending), events)?;
        Ok((growth.max(conversion), converted))
    }

    /// Checks merging `events` field events into one buffer beside every
    /// buffer, their sources included.
    #[cfg(feature = "field-inline")]
    fn merge_fields(&self, events: usize) -> Result<usize, TraceAllocationError> {
        let phase = TraceAllocationPhase::FieldMerge;
        let fields = bytes_of::<FieldEvent>(events, phase)?;
        self.others().check_request(phase, self.held(), fields)
    }
}

/// An accumulating trace's next tick.
enum Next {
    Tick(Tick),
    /// The pending cycles must become rows before this tick, which emits this.
    Convert(Emission),
}

/// An admitted tick of an accumulating trace.
struct Tick {
    capacity: usize,
    /// The emission that admitted it, or `None` for [`TICK_BOUND`].
    peeked: Option<Emission>,
    /// Bytes its conversion needs beyond the live buffers, which its advice
    /// writes may not take.
    reserved: usize,
}

impl LimitedTrace {
    /// Makes the admitted reservations of `rows` rows and a `window`-cycle
    /// pending window beside `resident` bytes.
    pub(super) fn reserve(
        limit: usize,
        resident: usize,
        rows: usize,
        window: usize,
    ) -> Result<Self, TraceAllocationError> {
        let phase = TraceAllocationPhase::Initial;
        let mut row_buffer = Vec::new();
        reserve_exact(&mut row_buffer, rows, phase)?;
        let mut cycles = Vec::new();
        reserve_exact(&mut cycles, window, phase)?;
        Ok(Self {
            limit,
            resident,
            cycles,
            rows: row_buffer,
            chunks: Vec::new(),
            field_bytes: 0,
        })
    }

    /// An accumulating trace with its `initial`-cycle buffer reserved, and the
    /// first tick's peeked emission if a peek admitted it, or `None` when
    /// [`Self::next_tick`] does not admit the first tick at that buffer.
    pub(super) fn accumulating(
        limit: usize,
        resident: usize,
        initial: usize,
        cpu: &mut Cpu,
    ) -> Result<Option<(Self, Option<Emission>)>, TraceError> {
        let mut trace = Self::reserve(limit, resident, 0, 0)?;
        let Next::Tick(tick) = trace.next_tick(cpu, 0, initial, None)? else {
            return Ok(None);
        };
        let phase = TraceAllocationPhase::Initial;
        reserve_exact(&mut trace.cycles, tick.capacity, phase)?;
        let emission = tick.peeked.unwrap_or(TICK_BOUND);
        let tape = cpu.advice_tape.capacity();
        let admitted = trace.admit_tick(trace.cycles.capacity(), emission, 0, tape);
        Ok(admitted.is_ok().then_some((trace, tick.peeked)))
    }

    /// Runs an accumulating trace to its end, returning `None`, or to a tick
    /// it cannot admit, returning that tick's emission unrun. `events` counts
    /// the cycles' field events. Advice writes leave room for the conversion
    /// that admitted their tick; a tick that changes the buffer's capacity
    /// fails with [`TraceError::UncountedCycles`].
    pub(super) fn accumulate(
        &mut self,
        emulator: &mut Emulator,
        events: &mut usize,
        mut peeked: Option<Emission>,
    ) -> Result<Option<Emission>, TraceError> {
        loop {
            let cpu = emulator.get_mut_cpu();
            let capacity = self.cycles.capacity();
            let tick = match self.next_tick(cpu, *events, capacity, peeked.take())? {
                Next::Tick(tick) => tick,
                Next::Convert(emission) => return Ok(Some(emission)),
            };
            let mut reserved = tick.reserved;
            if tick.capacity > capacity {
                let additional = tick.capacity - self.cycles.len();
                let phase = TraceAllocationPhase::CycleBuffer;
                reserve_exact(&mut self.cycles, additional, phase)?;
                // Admitted again at the capacity the allocator returned.
                let emission = tick.peeked.unwrap_or(TICK_BOUND);
                let tape = cpu.advice_tape.capacity();
                match self.admit_tick(self.cycles.capacity(), emission, *events, tape) {
                    Ok(bytes) => reserved = bytes,
                    Err(_) => return Ok(Some(emission)),
                }
            }
            cpu.set_advice_write_allowance(self.advice_write_allowance(reserved));
            let (pending, capacity) = (self.cycles.len(), self.cycles.capacity());
            crate::step_emulator(emulator, Some(&mut self.cycles));
            if let Some(refusal) = emulator.get_cpu().advice_write_refusal() {
                return Err(refusal.into());
            }
            if self.cycles.capacity() != capacity {
                return Err(TraceError::UncountedCycles { capacity });
            }
            if self.cycles.len() == pending {
                return Ok(None);
            }
            #[cfg(feature = "field-inline")]
            {
                *events += field_event_count(&self.cycles[pending..]);
            }
        }
    }

    /// Admits the next tick at a `capacity`-cycle buffer, whose cycles carry
    /// `events` field events: its cycles fit the buffer, grown by
    /// [`growth_target`] if they must, and converting every cycle after it
    /// fits beside the buffer. [`TICK_BOUND`] admits it without a peek where
    /// no growth is needed; else its emission is `peeked` or [`next_emission`].
    fn next_tick(
        &self,
        cpu: &mut Cpu,
        events: usize,
        capacity: usize,
        peeked: Option<Emission>,
    ) -> Result<Next, TraceError> {
        let tape = cpu.advice_tape.capacity();
        let pending = self.cycles.len();
        let emission = match peeked {
            Some(emission) => emission,
            None => {
                let bounded = pending
                    .checked_add(TICK_BOUND.rows)
                    .filter(|&rows| rows <= capacity)
                    .and_then(|_| self.admit_tick(capacity, TICK_BOUND, events, tape).ok());
                if let Some(reserved) = bounded {
                    let peeked = None;
                    return Ok(Next::Tick(Tick {
                        capacity,
                        peeked,
                        reserved,
                    }));
                }
                next_emission(cpu)?
            }
        };
        let phase = TraceAllocationPhase::CycleBuffer;
        let required = pending
            .checked_add(emission.rows)
            .ok_or(TraceAllocationError::SizeOverflow { phase })?;
        let target = match (required > capacity).then(|| growth_target(capacity, required)) {
            Some(target) => self.residency(tape).grow_cycles(target).map(|_| target),
            None => Ok(capacity),
        };
        let tick = target.and_then(|capacity| {
            let reserved = self.admit_tick(capacity, emission, events, tape)?;
            let peeked = Some(emission);
            Ok(Tick {
                capacity,
                peeked,
                reserved,
            })
        });
        Ok(tick.map_or(Next::Convert(emission), Next::Tick))
    }

    /// Checks converting every pending cycle after a tick that emits
    /// `emission`, the cycles before it carrying `events` field events, beside
    /// a `capacity`-cycle buffer. Returns the bytes the conversion needs
    /// beyond the buffers live before it.
    fn admit_tick(
        &self,
        capacity: usize,
        emission: Emission,
        events: usize,
        tape: usize,
    ) -> Result<usize, TraceAllocationError> {
        let overflow = TraceAllocationError::SizeOverflow {
            phase: TraceAllocationPhase::Conversion,
        };
        let pending = self
            .cycles
            .len()
            .checked_add(emission.rows)
            .ok_or(overflow)?;
        let events = events.checked_add(emission.field_events).ok_or(overflow)?;
        let cycles = bytes_of::<Cycle>(capacity, TraceAllocationPhase::CycleBuffer)?;
        let residency = Residency {
            cycles,
            ..self.residency(tape)
        };
        let (peak, _) = residency.plan_conversion(pending, events, self.row_target(pending)?)?;
        Ok(peak.saturating_sub(residency.live()))
    }

    fn residency(&self, tape: usize) -> Residency {
        Residency {
            limit: self.limit,
            resident: self.resident,
            tape,
            cycles: heap_bytes(&self.cycles),
            rows: heap_bytes(&self.rows),
            chunks: self.chunks.len(),
            chunk_capacity: self.chunks.capacity(),
            field_bytes: self.field_bytes,
        }
    }

    /// What an advice write may request beside every buffer of this trace and
    /// the `reserved` bytes a conversion still needs; the write's own check
    /// counts the advice tape.
    pub(super) fn advice_write_allowance(&self, reserved: usize) -> TraceAllocationAllowance {
        let live = [
            self.resident,
            heap_bytes(&self.cycles),
            heap_bytes(&self.rows),
            heap_bytes(&self.chunks),
            self.field_bytes,
            reserved,
        ]
        .into_iter()
        .fold(0, usize::saturating_add);
        TraceAllocationAllowance::new(self.limit, live)
    }

    /// The capacity converting `pending` more cycles grows the rows to by
    /// [`growth_target`], if they must grow.
    fn row_target(&self, pending: usize) -> Result<Option<usize>, TraceAllocationError> {
        let phase = TraceAllocationPhase::RowGrowth;
        let required = self
            .rows
            .len()
            .checked_add(pending)
            .ok_or(TraceAllocationError::SizeOverflow { phase })?;
        let capacity = self.rows.capacity();
        Ok((required > capacity).then(|| growth_target(capacity, required)))
    }

    /// Grows the rows to `capacity`, checked beside the old rows, the chunk
    /// metadata and every other buffer.
    fn grow_rows(&mut self, capacity: usize, tape: usize) -> Result<(), TraceAllocationError> {
        self.residency(tape).grow_rows(capacity)?;
        let additional = capacity - self.rows.len();
        reserve_exact(&mut self.rows, additional, TraceAllocationPhase::RowGrowth)
    }

    /// Replaces the empty pending window with one of `window` cycles, the old
    /// one dropped first, checked beside every other buffer.
    fn replace_window(&mut self, window: usize, tape: usize) -> Result<(), TraceAllocationError> {
        if self.cycles.capacity() != window {
            self.cycles = Vec::new();
            self.residency(tape).grow_cycles(window)?;
            reserve_exact(&mut self.cycles, window, TraceAllocationPhase::CycleBuffer)?;
        }
        Ok(())
    }

    /// Converts the accumulated cycles, which carry `events` field events,
    /// into rows of `anchor` capacity and leaves a one-tick window: straight
    /// into `anchor` rows if that fits beside the cycles, else into exact rows
    /// that grow to `anchor` once the cycles drop.
    pub(super) fn start_batches(
        &mut self,
        bytecode: &BytecodePreprocessing,
        tape: usize,
        anchor: usize,
        events: usize,
    ) -> Result<(), TraceError> {
        let pending = self.cycles.len();
        let direct = self
            .residency(tape)
            .plan_conversion(pending, events, Some(anchor));
        if direct.is_ok() {
            self.grow_rows(anchor, tape)?;
        }
        self.convert_pending(bytecode, tape)?;
        self.cycles = Vec::new();
        if self.rows.capacity() < anchor {
            self.grow_rows(anchor, tape)?;
        }
        self.replace_window(MAX_EXPANDED_ROWS_PER_SOURCE, tape)?;
        Ok(())
    }

    /// Reserves the next window at an empty one and returns its batch, the
    /// window less one tick: the longest window up to
    /// the domain-selected pending window whose cycles, field events and
    /// metadata fit beside every other buffer with the rows grown for it; else
    /// the longest within the rows' capacity; else one tick, for which rows
    /// that [`next_emission`] overruns grow while no window is live.
    pub(super) fn plan_batch(
        &mut self,
        cpu: &mut Cpu,
        batch_rows: usize,
    ) -> Result<usize, TraceError> {
        let tape = cpu.advice_tape.capacity();
        let phase = TraceAllocationPhase::CycleBuffer;
        let full =
            pending_window(batch_rows).ok_or(TraceAllocationError::SizeOverflow { phase })?;
        let residency = Residency {
            cycles: 0,
            rows: 0,
            ..self.residency(tape)
        };
        let (_, metadata) = residency.chunk_growth(conversion_slices(full))?;
        let others = bytes_of::<ConvertedChunk>(metadata.unwrap_or(0), phase)?
            .saturating_add(residency.live());
        let window_for = |rows: usize| {
            rows.checked_mul(size_of::<JoltTraceRow>())
                .and_then(|rows| rows.checked_add(others))
                .map_or(0, |held| {
                    self.limit.saturating_sub(held) / (size_of::<Cycle>() + FIELD_EVENT_BYTES)
                })
        };
        let (converted, capacity) = (self.rows.len(), self.rows.capacity());
        let required = converted.saturating_add(full);
        let growth = (required > capacity).then(|| growth_target(capacity, required));
        let mut window = full.min(window_for(capacity.saturating_add(growth.unwrap_or(0))));
        if window < MAX_EXPANDED_ROWS_PER_SOURCE {
            window = full.min(capacity - converted).min(window_for(capacity));
        }
        if window < MAX_EXPANDED_ROWS_PER_SOURCE {
            window = MAX_EXPANDED_ROWS_PER_SOURCE;
            if let Some(rows) = self.row_target(next_emission(cpu)?.rows)? {
                self.cycles = Vec::new();
                self.grow_rows(rows, tape)?;
            }
        }
        self.replace_window(window, tape)?;
        Ok(window - MAX_EXPANDED_ROWS_PER_SOURCE + 1)
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
        let parallel = self.cycles.len() > PARALLEL_ROW_CONVERSION_THRESHOLD;
        let _span = tracing::info_span!(
            "trace_rows_from_cycles",
            cycles = self.cycles.len(),
            mode = if parallel { "parallel" } else { "serial" }
        )
        .entered();
        if let Some(capacity) = self.row_target(self.cycles.len())? {
            self.grow_rows(capacity, tape)?;
        }
        let cycles = &self.cycles;
        let slices = conversion_slices(cycles.len());
        #[cfg(feature = "field-inline")]
        let events = field_event_count(cycles);
        #[cfg(not(feature = "field-inline"))]
        let events = 0;
        let residency = self.residency(tape);
        let (_, converted) = residency.convert(slices, events)?;
        let phase = TraceAllocationPhase::Conversion;
        if converted.chunk_capacity > self.chunks.capacity() {
            let additional = converted.chunk_capacity - self.chunks.len();
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
            let fields = self.chunks[first_chunk..]
                .iter()
                .map(|chunk| heap_bytes(&chunk.fields))
                .fold(0, usize::saturating_add);
            self.field_bytes = self.field_bytes.saturating_add(fields);
        }
        self.cycles.clear();
        Ok(())
    }

    /// The trace data, dimensions and final image of an accumulated trace,
    /// whose cycles carry `events` field events. When the image, and then,
    /// with the flat memory freed for it, the conversion and the field merge
    /// all fit, the image comes first, as in the original trace, and the flat
    /// memory drops before the conversion; otherwise the rows come first and
    /// the image last. Either way the cycles drop before the field merge.
    pub(super) fn finish_accumulated(
        mut self,
        bytecode: &BytecodePreprocessing,
        final_memory: Memory,
        tape: usize,
        events: usize,
    ) -> Result<(TraceData, ExecutionDimensions, MemoryImage), TraceError> {
        let image = final_memory.flat_image();
        let residency = self.residency(tape);
        let everything = TraceAllocationAllowance::new(self.limit, residency.live());
        let flat = final_memory.flat_heap_bytes();
        let pending = self.cycles.len();
        let image_first = image.bytes().and_then(|bytes| {
            everything.check_request(TraceAllocationPhase::FinalMemory, 0, bytes)?;
            let resident = residency
                .resident
                .saturating_sub(flat)
                .saturating_add(bytes);
            let freed = Residency {
                resident,
                ..residency
            };
            let (_, converted) =
                freed.plan_conversion(pending, events, self.row_target(pending)?)?;
            #[cfg(feature = "field-inline")]
            Residency {
                cycles: 0,
                ..converted
            }
            .merge_fields(events)?;
            #[cfg(not(feature = "field-inline"))]
            let _ = converted;
            Ok(())
        });
        if image_first.is_ok() {
            let bytes = image.try_collect(everything)?;
            self.resident = self
                .resident
                .saturating_sub(flat)
                .saturating_add(heap_bytes(&bytes));
            drop(final_memory);
            self.convert_pending(bytecode, tape)?;
            self.cycles = Vec::new();
            return self.finish(FinalImage::Collected(bytes), tape);
        }
        self.convert_pending(bytecode, tape)?;
        self.cycles = Vec::new();
        self.finish(FinalImage::Flat(image), tape)
    }

    /// The trace data, its dimensions and the final memory image. The merged
    /// field events are checked with every source buffer, the pending window
    /// and `tape` bytes of advice tape live; the image after the window drops,
    /// beside the trace data, the tape and the resident buffers.
    pub(super) fn finish(
        self,
        image: FinalImage<'_>,
        tape: usize,
    ) -> Result<(TraceData, ExecutionDimensions, MemoryImage), TraceError> {
        #[cfg(feature = "field-inline")]
        let fields = {
            let phase = TraceAllocationPhase::FieldMerge;
            let count = self.chunks.iter().map(|chunk| chunk.fields.len()).sum();
            self.residency(tape).merge_fields(count)?;
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
        let bytes = match image {
            FinalImage::Collected(bytes) => bytes,
            FinalImage::Flat(image) => {
                image.try_collect(TraceAllocationAllowance::new(limit, live))?
            }
        };
        Ok((data, dimensions, MemoryImage { bytes }))
    }
}

/// A limited trace's final memory image: collected before the last
/// conversion, or still to collect from the flat memory.
pub(super) enum FinalImage<'a> {
    Collected(Vec<(u64, u8)>),
    Flat(FlatImage<'a>),
}
