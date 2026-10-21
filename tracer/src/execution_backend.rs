use std::mem::MaybeUninit;
use std::{path::PathBuf, sync::Arc};

use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, ExecutionDimensions, ExecutionSummary, JoltProgram,
    MemoryImage, RamAddressBounds, TraceAllocationError, TraceAllocationPhase, TraceData,
    TraceError, TraceInputs, TraceOutput,
};
use jolt_program::expand::MAX_EXPANDED_ROWS_PER_SOURCE;
use jolt_program::preprocess::BytecodePCMapper;
#[cfg(feature = "field-inline")]
use jolt_program::{execution::FieldEvent, field_inline::FieldInlineTraceData};
use jolt_riscv::{CircuitFlags, JoltTraceRow};
use rayon::prelude::*;

use common::jolt_device::{JoltDevice, MemoryLayout};

use crate::emulator::cpu::{AdviceTape, Cpu};
use crate::emulator::decode_cache::DecodeCache;
use crate::emulator::memory::{flat_doublewords, FlatImage, Memory};
use crate::emulator::{Emulator, TEST_MEMORY_CAPACITY};
#[cfg(feature = "field-inline")]
use crate::instruction::RISCVCycle;
use crate::instruction::{Cycle, Emission};
use crate::parallel::{ChunkCheckpoint, ChunkWorker, PassOne, SnapshotPool};
use crate::trace_allocation::{
    bytes_of, doubling_target, growth_target, reserve_exact, sum_bytes, TraceAllocationAllowance,
};
use crate::trace_row::cycle_to_trace_row;

#[derive(Default, Debug, Clone)]
pub struct TracerBackend {
    pub elf_path: Option<PathBuf>,
    trace_allocation_limit: Option<usize>,
}

impl TracerBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_elf_path(elf_path: PathBuf) -> Self {
        Self {
            elf_path: Some(elf_path),
            ..Self::default()
        }
    }

    /// A backend whose [`ExecutionBackend::trace`] checks selected
    /// trace-storage requests against `limit_bytes`, or `None` when the
    /// original full-cycle trace's row payload over `row_domain` rows fits in
    /// `limit_bytes`; callers then keep their unlimited backend.
    ///
    /// The payload is `D = size_of::<Cycle>() * max(R0, N) +
    /// size_of::<JoltTraceRow>() * N` for `N = row_domain` and the initial cycle
    /// reserve `R0` (`JOLT_TRACER_CAPACITY_ROWS`, default 2^24 rows): one cycle
    /// backing of at least `max(R0, N)` entries beside `N` rows. `D` at or
    /// below `limit_bytes` keeps the original trace, including its
    /// `TRACER_PARALLEL` opt-in; `D` above it, or overflowing, selects the
    /// limited serial trace.
    ///
    /// `D` is a routing proxy for a configured domain, not a measured demand or
    /// a peak: `N` neither counts nor caps the rows a run emits, and the `R0`
    /// reserve counts although it is mostly unfaulted. A run much shorter than
    /// `N`, or one dominated by the `R0` reserve, can take the limited trace
    /// even when the original trace would fit. A limit at or just above `D`
    /// keeps the original trace even when its unpriced requests exceed the
    /// limit: the cycle vector's growth past `R0`, which holds the old and new
    /// backings together (by itself above `D` when a field-inline run fills
    /// a domain larger than `R0`), the final memory image, the pristine
    /// emulator clone, flat guest memory and the PC map.
    ///
    /// The limited trace checks before each covered request that the covered
    /// buffers at observed capacity plus the request stay within
    /// `limit_bytes`; otherwise it returns [`TraceError::TraceAllocation`] and
    /// no output. Covered: the bytecode PC map, flat guest memory, the ELF copy
    /// made during emulator setup, the caller's input and advice buffers and
    /// the device's copies, the output reserve, the cycles and rows and their
    /// growth, a pending window of one full tick past the batch, converted-chunk
    /// metadata and field events and their merge, the advice tape and each host
    /// advice write's guest-copy scratch and tape growth, and the final memory
    /// image. Not covered: ELF headers and symbols, decode and expansion
    /// caches, inline advice and provider temporaries, cycle markers, prints
    /// and backtraces, the caller's program and preprocessing, and allocator
    /// and runtime overhead. The limit bounds requested payload, not process
    /// memory or whether a proof of the trace fits, and a rejected trace's
    /// guest effects are discarded, not rolled back. A limit below the setup
    /// requests, which exceed 1 GiB with the default `R0`, fails before setup,
    /// and the output keeps its rows' observed capacity: a run holding every
    /// cycle to the end requests exactly its rows; a batched run keeps at least
    /// `R0` rows, even when it converts only at the end.
    ///
    /// Under the limit, `TRACER_PARALLEL` of 2 or more and
    /// [`ChunkedExecutionBackend::execute`] fail with
    /// [`TraceError::UnsupportedTraceAllocationMode`].
    /// [`ChunkedExecutionBackend::replay_chunk`] is outside the limit: it
    /// replays a checkpoint without checking any request.
    pub fn limited_for_row_domain(limit_bytes: usize, row_domain: usize) -> Option<Self> {
        let payload = size_of::<Cycle>()
            .checked_mul(crate::trace_capacity_reserve().max(row_domain))
            .and_then(|cycles| {
                cycles.checked_add(size_of::<JoltTraceRow>().checked_mul(row_domain)?)
            });
        payload
            .is_none_or(|payload| payload > limit_bytes)
            .then(|| Self {
                trace_allocation_limit: Some(limit_bytes),
                ..Self::default()
            })
    }

    fn trace_execution(
        &self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceExecution, TraceError> {
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }

        let (_lazy_trace, cycles, final_memory, device, advice_tape) = crate::trace(
            program.elf_bytes(),
            self.elf_path.as_ref(),
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            inputs.advice_tape.map(AdviceTape::from_bytes),
        );
        Ok(TraceExecution {
            cycles,
            final_memory: MemoryImage {
                bytes: final_memory.materialized_nonzero_bytes(),
            },
            device,
            advice_tape: advice_tape.into_bytes(),
        })
    }

    /// [`ExecutionBackend::trace`] under `limit`, serially. While converting
    /// every cycle after the next tick fits beside them, the run keeps its
    /// cycles in one buffer, `R0` at first and grown by [`growth_target`], and
    /// converts them once, at the end. Before a tick for which that or the
    /// growth does not fit, it converts them and batches, replanning its window
    /// after each conversion. When the setup or the first tick cannot hold `R0`
    /// cycles, it batches from the start: the pending cycles become rows after
    /// each tick that leaves at least `batch_rows` of them, and at the end.
    #[tracing::instrument(skip_all, name = "trace")]
    fn trace_limited(
        &self,
        program: &JoltProgram,
        inputs: TraceInputs,
        limit: usize,
        batch_rows: usize,
    ) -> Result<TraceOutput, TraceError> {
        if crate::parallel_config_from_env().is_some() {
            return Err(TraceError::UnsupportedTraceAllocationMode);
        }
        let phase = TraceAllocationPhase::Initial;
        let initial = crate::trace_capacity_reserve();
        let window =
            pending_window(batch_rows).ok_or(TraceAllocationError::SizeOverflow { phase })?;
        let pc_map = BytecodePCMapper::required_allocation_bytes(&program.expanded_bytecode)?;
        let storage = sum_bytes(
            [
                pc_map,
                bytes_of::<JoltTraceRow>(initial, phase)?,
                bytes_of::<Cycle>(window, phase)?,
            ],
            phase,
        )?;
        // Both admissions precede setup, which moves the caller's advice tape
        // they count.
        let batched = LimitedSetup::admit(limit, program, &inputs, storage);
        let accumulating = bytes_of::<Cycle>(initial, phase)
            .and_then(|cycles| sum_bytes([pc_map, cycles], phase))
            .map_err(TraceError::from)
            .and_then(|storage| LimitedSetup::admit(limit, program, &inputs, storage));
        let (setup, batched) = match accumulating {
            Ok(setup) => (setup, Some(batched)),
            Err(_) => (batched?, None),
        };
        let bytecode = BytecodePCMapper::try_new(&program.expanded_bytecode)?;
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        let mut emulator = crate::create_emulator(
            program.elf_bytes(),
            self.elf_path.as_ref(),
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            inputs.advice_tape.map(AdviceTape::from_bytes),
        );
        let resident = setup
            .reserve_device(&mut emulator)?
            .saturating_add(bytecode.heap_bytes());
        let mut prev_pc = 0;
        let accumulated = match batched {
            Some(batched) => {
                let cpu = emulator.get_mut_cpu();
                let first = LimitedTrace::accumulating(limit, resident, initial, cpu)?;
                if first.is_none() {
                    batched?;
                }
                first
            }
            None => None,
        };
        // The field events of a run that accumulated every cycle; a run that
        // accumulated first plans each batch after its transition.
        let mut accumulated_events = None;
        let (mut trace, mut batch_rows, planned) = match accumulated {
            Some((mut trace, peeked)) => {
                let mut events = 0;
                match trace.accumulate(&mut emulator, &mut prev_pc, &mut events, peeked)? {
                    Some(next) => {
                        let phase = TraceAllocationPhase::RowGrowth;
                        let anchor = trace
                            .cycles
                            .len()
                            .checked_add(next.rows)
                            .and_then(|rows| doubling_target(initial, rows))
                            .ok_or(TraceAllocationError::SizeOverflow { phase })?;
                        let tape = emulator.get_cpu().advice_tape.capacity();
                        trace.start_batches(&bytecode, tape, anchor, events)?;
                    }
                    None => accumulated_events = Some(events),
                }
                (trace, 1, true)
            }
            None => {
                let trace = LimitedTrace::reserve(limit, resident, initial, window)?;
                (trace, batch_rows, false)
            }
        };
        let cpu = emulator.get_mut_cpu();
        cpu.set_advice_write_allowance(trace.advice_write_allowance(0));
        while accumulated_events.is_none() {
            let pending = trace.cycles.len();
            crate::step_emulator(&mut emulator, &mut prev_pc, Some(&mut trace.cycles));
            if let Some(refusal) = emulator.get_cpu().advice_write_refusal() {
                return Err(refusal.into());
            }
            // As in `crate::trace`, the first step that emits no row ends the run.
            if trace.cycles.len() == pending {
                break;
            }
            if trace.cycles.len() >= batch_rows {
                let tape = emulator.get_cpu().advice_tape.capacity();
                trace.convert_pending(&bytecode, tape)?;
                let cpu = emulator.get_mut_cpu();
                if planned {
                    batch_rows = trace.plan_batch(cpu, prev_pc)?;
                }
                cpu.set_advice_write_allowance(trace.advice_write_allowance(0));
            }
        }
        let (advice_tape, final_memory, device) = crate::finish_emulator(emulator);
        let tape = advice_tape.capacity();
        let (data, dimensions, image) = match accumulated_events {
            Some(events) => trace.finish_accumulated(&bytecode, final_memory, tape, events)?,
            None => {
                trace.convert_pending(&bytecode, tape)?;
                trace.finish(FinalImage::Flat(final_memory.flat_image()), tape)?
            }
        };
        Ok(TraceOutput::with_dimensions(
            dimensions,
            Arc::new(data),
            device,
            Some(image),
            Some(advice_tape.into_bytes()),
        ))
    }

    /// The advice tape `program` writes in one serial run without a trace,
    /// with its requests checked against `limit_bytes`: the compute-advice run
    /// the SDK makes under a memory budget.
    ///
    /// The run allocates no cycles, rows, PC map or final memory image. Before
    /// setup, the device's input copies and output reserve, flat guest memory
    /// and the ELF copy setup makes are checked beside the caller's buffers
    /// and advice tape, as for [`Self::limited_for_row_domain`]'s trace, and
    /// each host advice write's guest-copy scratch and tape growth beside the
    /// tape and those buffers. A refused request returns
    /// [`TraceError::TraceAllocation`] and no tape; the guest effects before
    /// it are discarded, not rolled back. The run ends as a trace does, at a
    /// PC stall or at the first step that emits no row, and does not read
    /// `TRACER_PARALLEL`. Not covered: decode and expansion caches, inline
    /// advice and provider temporaries, cycle markers, prints and backtraces,
    /// the caller's program, and allocator and runtime overhead; the limit
    /// bounds requested payload, not process memory.
    pub fn execute_advice_tape(
        program: &JoltProgram,
        inputs: TraceInputs,
        limit_bytes: usize,
    ) -> Result<Vec<u8>, TraceError> {
        let setup = LimitedSetup::admit(limit_bytes, program, &inputs, 0)?;
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        let mut emulator = crate::create_emulator(
            program.elf_bytes(),
            None,
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            inputs.advice_tape.map(AdviceTape::from_bytes),
        );
        let resident = setup.reserve_device(&mut emulator)?;
        let cpu = emulator.get_mut_cpu();
        cpu.set_advice_write_allowance(TraceAllocationAllowance::new(limit_bytes, resident));
        let mut pass = PassOne::new(emulator);
        loop {
            let running = pass.step();
            if let Some(refusal) = pass.emulator().get_cpu().advice_write_refusal() {
                return Err(refusal.into());
            }
            if !running {
                break;
            }
        }
        let (advice_tape, _, _) = crate::finish_emulator(pass.into_emulator());
        Ok(advice_tape.into_bytes())
    }
}

struct TraceExecution {
    cycles: Vec<Cycle>,
    final_memory: MemoryImage,
    device: JoltDevice,
    advice_tape: Vec<u8>,
}

impl ExecutionBackend for TracerBackend {
    type Trace = Arc<TraceData>;

    fn trace(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput, TraceError> {
        if let Some(limit) = self.trace_allocation_limit {
            return self.trace_limited(program, inputs, limit, SERIAL_BATCH_ROWS);
        }
        let bytecode = BytecodePCMapper::try_new(&program.expanded_bytecode)?;
        let execution = self.trace_execution(program, inputs)?;
        let (data, dimensions) = collect_rows(&execution.cycles, &bytecode)?;
        Ok(TraceOutput::with_dimensions(
            dimensions,
            Arc::new(data),
            execution.device,
            Some(execution.final_memory),
            Some(execution.advice_tape),
        ))
    }
}

const PARALLEL_ROW_CONVERSION_THRESHOLD: usize = 1 << 14;

/// Pending cycles at which a limited trace that batches from the start
/// converts a batch, and the most a planned batch holds: 96 MiB of
/// default-build cycles, 64 parallel conversion slices.
const SERIAL_BATCH_ROWS: usize = 1 << 20;

/// Cycles a limited trace can hold pending: each tick starts with fewer than
/// `batch_rows` and emits one source instruction's expansion at most.
fn pending_window(batch_rows: usize) -> Option<usize> {
    batch_rows
        .max(1)
        .checked_add(MAX_EXPANDED_ROWS_PER_SOURCE - 1)
}

fn heap_bytes<T>(buffer: &Vec<T>) -> usize {
    buffer.capacity() * size_of::<T>()
}

#[cfg(feature = "field-inline")]
fn field_event_count(cycles: &[Cycle]) -> usize {
    cycles
        .iter()
        .filter(|cycle| cycle.field_inline_trace().is_some())
        .count()
}

#[cfg(feature = "field-inline")]
const FIELD_EVENT_BYTES: usize = size_of::<FieldEvent>();
#[cfg(not(feature = "field-inline"))]
const FIELD_EVENT_BYTES: usize = 0;

/// The most rows, and field events, one tick emits.
const TICK_BOUND: Emission = Emission {
    rows: MAX_EXPANDED_ROWS_PER_SOURCE,
    field_events: MAX_EXPANDED_ROWS_PER_SOURCE,
};

/// What the next tick emits: nothing at a PC stall, while the CPU waits for
/// an interrupt or where its fetch traps, else its instruction's emission.
fn next_emission(cpu: &mut Cpu, prev_pc: u64) -> Result<Emission, TraceError> {
    if prev_pc == cpu.read_pc() {
        return Ok(Emission::default());
    }
    match cpu.peek_instruction() {
        Some(instruction) => instruction.emission(cpu),
        None => Ok(Emission::default()),
    }
}

/// The parallel slices, or one serial slice, a conversion of `pending` cycles
/// fills: one converted chunk each.
fn conversion_slices(pending: usize) -> usize {
    if pending > PARALLEL_ROW_CONVERSION_THRESHOLD {
        pending.div_ceil(PARALLEL_ROW_CONVERSION_THRESHOLD)
    } else {
        1
    }
}

#[derive(Default)]
struct ConvertedChunk {
    bounds: RamAddressBounds,
    #[cfg(feature = "field-inline")]
    fields: Vec<FieldEvent>,
}

impl ConvertedChunk {
    /// Writes the rows of `cycles`, the trace's cycles from index `first`,
    /// into `slots`, and records their RAM bounds and field events.
    fn fill(
        &mut self,
        first: usize,
        slots: &mut [MaybeUninit<JoltTraceRow>],
        cycles: &[Cycle],
        bytecode: &BytecodePCMapper,
    ) -> Result<(), TraceError> {
        #[cfg(not(feature = "field-inline"))]
        let _ = first;
        for (offset, (slot, cycle)) in slots.iter_mut().zip(cycles).enumerate() {
            #[cfg(not(feature = "field-inline"))]
            let _ = offset;
            let row = cycle_to_trace_row(cycle, bytecode)?;
            self.bounds.observe(row.ram_address());
            slot.write(row);
            #[cfg(feature = "field-inline")]
            if let Some(data) = cycle.field_inline_trace() {
                self.fields.push(FieldEvent {
                    cycle: first + offset,
                    data,
                });
            }
        }
        Ok(())
    }
}

fn collect_rows(
    cycles: &[Cycle],
    bytecode: &BytecodePCMapper,
) -> Result<(TraceData, ExecutionDimensions), TraceError> {
    let parallel = cycles.len() > PARALLEL_ROW_CONVERSION_THRESHOLD;
    let _span = tracing::info_span!(
        "trace_rows_from_cycles",
        cycles = cycles.len(),
        mode = if parallel { "parallel" } else { "serial" }
    )
    .entered();
    let mut rows = Vec::<JoltTraceRow>::with_capacity(cycles.len());
    let slots = &mut rows.spare_capacity_mut()[..cycles.len()];
    let fill = |batch: usize, slots: &mut [MaybeUninit<JoltTraceRow>], cycles: &[Cycle]| {
        let mut chunk = ConvertedChunk::default();
        chunk.fill(
            batch * PARALLEL_ROW_CONVERSION_THRESHOLD,
            slots,
            cycles,
            bytecode,
        )?;
        Ok::<_, TraceError>(chunk)
    };
    let batches = if parallel {
        slots
            .par_chunks_mut(PARALLEL_ROW_CONVERSION_THRESHOLD)
            .zip(cycles.par_chunks(PARALLEL_ROW_CONVERSION_THRESHOLD))
            .enumerate()
            .map(|(batch, (slots, cycles))| fill(batch, slots, cycles))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![fill(0, slots, cycles)?]
    };
    // SAFETY: successful workers initialized every slot. JoltTraceRow is Copy,
    // so partially written rows need no cleanup on error or unwinding.
    unsafe { rows.set_len(cycles.len()) };
    #[cfg(feature = "field-inline")]
    let fields = Vec::with_capacity(batches.iter().map(|batch| batch.fields.len()).sum());
    finish_rows(
        rows,
        batches,
        #[cfg(feature = "field-inline")]
        fields,
    )
}

/// The trace data and dimensions of `rows`, which `chunks` converted in
/// order. `fields`, empty and with room for every chunk's events, receives
/// them merged.
fn finish_rows(
    rows: Vec<JoltTraceRow>,
    chunks: Vec<ConvertedChunk>,
    #[cfg(feature = "field-inline")] mut fields: Vec<FieldEvent>,
) -> Result<(TraceData, ExecutionDimensions), TraceError> {
    let ram_bounds = chunks
        .iter()
        .fold(RamAddressBounds::default(), |bounds, chunk| {
            bounds.merge(chunk.bounds)
        });
    #[cfg(not(feature = "field-inline"))]
    let data = TraceData::new(rows);
    #[cfg(feature = "field-inline")]
    let data = {
        for chunk in chunks {
            fields.extend(chunk.fields);
        }
        TraceData::from_parts(rows, fields)?
    };
    let dimensions = ExecutionDimensions {
        trace_length: data.proof_len(),
        ends_in_jump: data
            .proof_rows()
            .last()
            .is_some_and(|row| row.circuit_flags().get(CircuitFlags::Jump)),
        ram_bounds,
    };
    Ok((data, dimensions))
}

/// Setup requests of a limited run, admitted together with the run's own
/// setup storage before any of them is made.
struct LimitedSetup {
    output_bytes: usize,
    /// The caller's input and advice buffers, which outlive the run.
    caller_bytes: usize,
}

impl LimitedSetup {
    /// Checks the device's input copies and output reserve, flat guest
    /// memory, the ELF copy setup makes and `storage` bytes of the run's own
    /// setup requests beside the caller's buffers and advice tape.
    fn admit(
        limit: usize,
        program: &JoltProgram,
        inputs: &TraceInputs,
        storage: usize,
    ) -> Result<Self, TraceError> {
        let phase = TraceAllocationPhase::Initial;
        let overflow = TraceAllocationError::SizeOverflow { phase };
        let layout = MemoryLayout::try_new(&inputs.memory_config)?;
        let output_bytes = usize::try_from(layout.max_output_size).map_err(|_| overflow)?;
        // `setup_program` sizes a riscv-tests ELF (one with `tohost`) by the
        // test capacity instead of the layout.
        let memory_words =
            flat_doublewords(layout.get_total_memory_size().max(TEST_MEMORY_CAPACITY));
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
        let held = caller_bytes.saturating_add(tape_bytes);
        TraceAllocationAllowance::new(limit, 0).check_request(phase, held, requests)?;
        Ok(Self {
            output_bytes,
            caller_bytes,
        })
    }

    /// Makes the admitted output reservation and returns the bytes the
    /// caller's buffers, the device's input copies and outputs and flat guest
    /// memory hold at their observed capacity.
    #[expect(clippy::expect_used, reason = "create_emulator installs the device")]
    fn reserve_device(&self, emulator: &mut Emulator) -> Result<usize, TraceAllocationError> {
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
struct LimitedTrace {
    limit: usize,
    /// The PC map, the caller's buffers, the device's input copies and output
    /// reserve, and flat guest memory (or the final image collected in its
    /// place), none of which grows after setup.
    resident: usize,
    /// The pending cycles: every cycle while the trace accumulates, otherwise
    /// a window that holds one full tick past the batch.
    cycles: Vec<Cycle>,
    rows: Vec<JoltTraceRow>,
    chunks: Vec<ConvertedChunk>,
    /// Bytes of the converted chunks' field-event buffers.
    fields: usize,
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
    fields: usize,
}

impl Residency {
    /// What a request may take beside the resident buffers, the cycles, the
    /// chunks' field events and the advice tape.
    fn others(&self) -> TraceAllocationAllowance {
        let live = [self.resident, self.cycles, self.fields, self.tape]
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
        let others = [self.resident, self.cycles, self.fields, self.tape];
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
            fields: self.fields.saturating_add(fields),
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
    fn reserve(
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
            fields: 0,
        })
    }

    /// An accumulating trace with its `initial`-cycle buffer reserved, and the
    /// first tick's peeked emission if a peek admitted it, or `None` when
    /// [`Self::next_tick`] does not admit the first tick at that buffer.
    fn accumulating(
        limit: usize,
        resident: usize,
        initial: usize,
        cpu: &mut Cpu,
    ) -> Result<Option<(Self, Option<Emission>)>, TraceError> {
        let mut trace = Self::reserve(limit, resident, 0, 0)?;
        let Next::Tick(tick) = trace.next_tick(cpu, 0, 0, initial, None)? else {
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
    fn accumulate(
        &mut self,
        emulator: &mut Emulator,
        prev_pc: &mut u64,
        events: &mut usize,
        mut peeked: Option<Emission>,
    ) -> Result<Option<Emission>, TraceError> {
        loop {
            let cpu = emulator.get_mut_cpu();
            let capacity = self.cycles.capacity();
            let tick = match self.next_tick(cpu, *prev_pc, *events, capacity, peeked.take())? {
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
            crate::step_emulator(emulator, prev_pc, Some(&mut self.cycles));
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
        prev_pc: u64,
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
                next_emission(cpu, prev_pc)?
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
            fields: self.fields,
        }
    }

    /// What an advice write may request beside every buffer of this trace and
    /// the `reserved` bytes a conversion still needs; the write's own check
    /// counts the advice tape.
    fn advice_write_allowance(&self, reserved: usize) -> TraceAllocationAllowance {
        let live = [
            self.resident,
            heap_bytes(&self.cycles),
            heap_bytes(&self.rows),
            heap_bytes(&self.chunks),
            self.fields,
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
    fn start_batches(
        &mut self,
        bytecode: &BytecodePCMapper,
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
    /// `pending_window(SERIAL_BATCH_ROWS)` whose cycles, field events and
    /// metadata fit beside every other buffer with the rows grown for it; else
    /// the longest within the rows' capacity; else one tick, for which rows
    /// that [`next_emission`] overruns grow while no window is live.
    fn plan_batch(&mut self, cpu: &mut Cpu, prev_pc: u64) -> Result<usize, TraceError> {
        let tape = cpu.advice_tape.capacity();
        let phase = TraceAllocationPhase::CycleBuffer;
        let full = pending_window(SERIAL_BATCH_ROWS)
            .ok_or(TraceAllocationError::SizeOverflow { phase })?;
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
            if let Some(rows) = self.row_target(next_emission(cpu, prev_pc)?.rows)? {
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
    fn convert_pending(
        &mut self,
        bytecode: &BytecodePCMapper,
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
            self.fields = self.fields.saturating_add(fields);
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
    fn finish_accumulated(
        mut self,
        bytecode: &BytecodePCMapper,
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
    fn finish(
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
enum FinalImage<'a> {
    Collected(Vec<(u64, u8)>),
    Flat(FlatImage<'a>),
}

/// A resume point for the chunked-execution contract, built on the two-pass
/// parallel machinery (PR #1717): a tick-boundary checkpoint and memory
/// image — captured together at the same boundary and shared, via `Arc`,
/// by every chunk that resumes there — plus this chunk's row window
/// relative to that boundary.
pub struct TracerChunkCheckpoint {
    boundary: Arc<ChunkCheckpoint>,
    /// Full-size flat-memory image at the same boundary (SnapshotPool
    /// layout).
    image: Arc<Vec<u64>>,
    seed: Arc<WorkerSeed>,
    skip_rows: usize,
    take_rows: usize,
}

struct WorkerSeed {
    device: Option<JoltDevice>,
    decode: DecodeCache,
    bytecode: BytecodePCMapper,
}

/// Boundary checkpoints are captured at chunk-mark crossings, but at most
/// one per this many rows: each carries a full-size memory image, so denser
/// capture (e.g. a small chunk size over a long trace) would blow up
/// memory. `skip_rows` absorbs the gap; per-chunk replay cost stays bounded
/// by spacing + chunk_size + one tick's rows.
const MIN_BOUNDARY_SPACING_ROWS: usize = 1 << 16;

impl TracerBackend {
    /// [`ChunkedExecutionBackend::execute`] with an explicit boundary
    /// spacing floor. The trait method passes [`MIN_BOUNDARY_SPACING_ROWS`];
    /// tests pass a tighter floor to exercise multi-boundary selection on
    /// guests whose whole trace is shorter than the production floor.
    fn execute_chunked(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
        chunk_size: usize,
        min_boundary_spacing: usize,
    ) -> Result<ExecutionSummary<TracerChunkCheckpoint>, TraceError> {
        if self.trace_allocation_limit.is_some() {
            return Err(TraceError::UnsupportedTraceAllocationMode);
        }
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        if chunk_size == 0 {
            return Err(TraceError::Backend("chunk_size must be nonzero"));
        }

        let bytecode = BytecodePCMapper::try_new(&program.expanded_bytecode)?;
        let emulator = crate::create_emulator(
            program.elf_bytes(),
            self.elf_path.as_ref(),
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            inputs.advice_tape.map(AdviceTape::from_bytes),
        );
        let seed = Arc::new(WorkerSeed {
            device: emulator.get_cpu().mmu.jolt_device.clone(),
            decode: emulator
                .get_cpu()
                .mmu
                .decode_cache
                .snapshot_with_empty_entries(),
            bytecode,
        });

        struct Boundary {
            checkpoint: Arc<ChunkCheckpoint>,
            image: Arc<Vec<u64>>,
            rows: usize,
        }

        let mut pool = SnapshotPool::new();
        let mut pass = PassOne::new(emulator);
        let capture = |pass: &PassOne, pool: &mut SnapshotPool| Boundary {
            checkpoint: Arc::new(pass.checkpoint()),
            image: Arc::new(pool.capture(&pass.emulator().get_cpu().mmu.memory.memory)),
            rows: pass.rows(),
        };

        let mut boundaries = vec![capture(&pass, &mut pool)];
        let mut next_mark = chunk_size;
        while pass.step() {
            if pass.rows() >= next_mark {
                #[expect(clippy::expect_used)]
                let last_rows = boundaries.last().expect("initial checkpoint present").rows;
                if pass.rows() - last_rows >= min_boundary_spacing {
                    boundaries.push(capture(&pass, &mut pool));
                }
                next_mark = (pass.rows() / chunk_size + 1) * chunk_size;
            }
        }
        let trace_len = pass.rows();

        let mut checkpoints = Vec::with_capacity(trace_len.div_ceil(chunk_size));
        let mut boundary_index = 0;
        for chunk in 0..trace_len.div_ceil(chunk_size) {
            let mark = chunk * chunk_size;
            while boundary_index + 1 < boundaries.len()
                && boundaries[boundary_index + 1].rows <= mark
            {
                boundary_index += 1;
            }
            let boundary = &boundaries[boundary_index];
            checkpoints.push(TracerChunkCheckpoint {
                boundary: Arc::clone(&boundary.checkpoint),
                image: Arc::clone(&boundary.image),
                seed: Arc::clone(&seed),
                skip_rows: mark - boundary.rows,
                take_rows: chunk_size.min(trace_len - mark),
            });
        }

        let (advice_tape, final_memory, device) = crate::finish_emulator(pass.into_emulator());

        Ok(ExecutionSummary {
            checkpoints,
            trace_len,
            device,
            final_memory: Some(MemoryImage {
                bytes: final_memory.materialized_nonzero_bytes(),
            }),
            advice_tape: Some(advice_tape.into_bytes()),
        })
    }
}

impl ChunkedExecutionBackend for TracerBackend {
    type Checkpoint = TracerChunkCheckpoint;

    fn execute(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
        chunk_size: usize,
    ) -> Result<ExecutionSummary<Self::Checkpoint>, TraceError> {
        self.execute_chunked(program, inputs, chunk_size, MIN_BOUNDARY_SPACING_ROWS)
    }

    fn replay_chunk(&self, checkpoint: &Self::Checkpoint) -> Result<TraceData, TraceError> {
        let mut worker = ChunkWorker::from_seed(
            checkpoint.seed.device.clone(),
            checkpoint.seed.decode.clone(),
        );
        let _previous =
            worker.install_chunk(&checkpoint.boundary, checkpoint.image.as_ref().clone());

        let needed = checkpoint.skip_rows + checkpoint.take_rows;
        let mut cycles: Vec<Cycle> = Vec::with_capacity(needed + 64);
        while cycles.len() < needed {
            let before = cycles.len();
            worker.run_ticks(1, &mut cycles);
            if cycles.len() == before {
                // A zero-row tick (trap/WFI) before the window is complete:
                // fail instead of spinning forever. This detects stalls
                // only — a replay that diverges while still emitting rows
                // is not caught here. run_two_pass's per-chunk row-count
                // and boundary-state tripwires don't transfer to
                // row-aligned contract checkpoints (no expected tick count
                // exists for a window ending mid-tick, and no boundary
                // state is captured at arbitrary marks); count- and
                // value-level fidelity is instead enforced by the
                // chunk-composition equivalence tests (invariant 3 of
                // specs/x86-tracer-backend.md).
                return Err(TraceError::Backend(
                    "chunk replay stalled before completing its row window",
                ));
            }
        }

        collect_rows(
            &cycles[checkpoint.skip_rows..needed],
            &checkpoint.seed.bytecode,
        )
        .map(|(data, _)| data)
    }
}

#[cfg(feature = "field-inline")]
impl Cycle {
    pub fn field_inline_trace(&self) -> Option<FieldInlineTraceData> {
        let register_state = match self {
            Self::FIELD_ADD(RISCVCycle { register_state, .. })
            | Self::FIELD_SUB(RISCVCycle { register_state, .. })
            | Self::FIELD_MUL(RISCVCycle { register_state, .. })
            | Self::FIELD_INV(RISCVCycle { register_state, .. })
            | Self::FIELD_ASSERT_EQ(RISCVCycle { register_state, .. })
            | Self::FIELD_ASSERT_ZERO(RISCVCycle { register_state, .. })
            | Self::FIELD_LOAD_ACCUMULATE_FROM_REGISTER(RISCVCycle { register_state, .. })
            | Self::FIELD_LOAD_ACCUMULATE_FROM_MEMORY(RISCVCycle { register_state, .. })
            | Self::FIELD_LOAD_IMM(RISCVCycle { register_state, .. })
            | Self::FIELD_ADVICE_LIMB(RISCVCycle { register_state, .. }) => register_state,
            _ => return None,
        };
        Some(register_state.to_field_inline_trace())
    }
}

#[cfg(test)]
mod chunked_tests {
    use super::*;
    use crate::test_utils::build_muldiv_guest;
    use common::jolt_device::MemoryConfig;
    use jolt_program::execution::build_jolt_program;

    const INPUTS: [u8; 6] = [0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c];

    fn muldiv_setup() -> (JoltProgram, TraceInputs) {
        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let program = build_jolt_program(&elf).expect("failed to build Jolt program");
        let inputs = TraceInputs::new(INPUTS.to_vec(), Vec::new(), Vec::new(), memory_config);
        (program, inputs)
    }

    /// Invariant 3 of `specs/x86-tracer-backend.md` for the reference backend:
    /// the concatenation of `replay_chunk` outputs equals the eager `trace()`
    /// row stream for every chunk size, including degenerate ones, regardless
    /// of replay order.
    #[test]
    fn chunked_execution_composes_to_eager_trace() {
        let (program, inputs) = muldiv_setup();
        let mut backend = TracerBackend::new();
        let eager = backend
            .trace(&program, inputs.clone())
            .expect("eager trace failed");
        let eager_rows = eager.trace.rows();
        assert!(!eager_rows.is_empty());

        // Chunk size 1 forces checkpoint marks inside multi-row expansions.
        // Spacing = chunk_size keeps boundary checkpoints dense enough to
        // exercise multi-boundary selection on a trace shorter than the
        // production floor (the floor itself is covered below).
        for chunk_size in [1usize, 100, 1 << 18, eager_rows.len() + 1] {
            let mut backend = TracerBackend::new();
            let summary = backend
                .execute_chunked(&program, inputs.clone(), chunk_size, chunk_size)
                .expect("execute failed");

            assert_eq!(
                summary.trace_len,
                eager_rows.len(),
                "chunk_size {chunk_size}"
            );
            assert_eq!(
                summary.checkpoints.len(),
                eager_rows.len().div_ceil(chunk_size),
                "chunk_size {chunk_size}"
            );
            assert_eq!(summary.device, eager.device, "chunk_size {chunk_size}");
            assert_eq!(
                summary.final_memory, eager.final_memory,
                "chunk_size {chunk_size}"
            );
            assert_eq!(
                summary.advice_tape, eager.advice_tape,
                "chunk_size {chunk_size}"
            );

            // Replay in reverse order to exercise order-independence.
            let mut replayed: Vec<Vec<JoltTraceRow>> = summary
                .checkpoints
                .iter()
                .rev()
                .map(|checkpoint| {
                    backend
                        .replay_chunk(checkpoint)
                        .expect("replay failed")
                        .rows()
                        .to_vec()
                })
                .collect();
            replayed.reverse();

            let last = replayed.len() - 1;
            for (i, rows) in replayed.iter().enumerate() {
                if i < last {
                    assert_eq!(rows.len(), chunk_size, "chunk_size {chunk_size}, chunk {i}");
                }
            }
            let concat: Vec<JoltTraceRow> = replayed.into_iter().flatten().collect();
            assert_eq!(concat.as_slice(), eager_rows, "chunk_size {chunk_size}");
        }
    }

    /// The public trait method applies the production spacing floor: the
    /// muldiv trace is shorter than [`MIN_BOUNDARY_SPACING_ROWS`], so every
    /// chunk resumes from the single initial checkpoint and `skip_rows`
    /// grows to nearly the whole trace for the last chunk.
    #[test]
    fn default_spacing_floor_replays_through_large_skips() {
        let (program, inputs) = muldiv_setup();
        let mut backend = TracerBackend::new();
        let eager = backend
            .trace(&program, inputs.clone())
            .expect("eager trace failed");
        let eager_rows = eager.trace.rows();

        const CHUNK_SIZE: usize = 100;
        assert!(eager_rows.len() > CHUNK_SIZE);
        assert!(
            eager_rows.len() < MIN_BOUNDARY_SPACING_ROWS,
            "muldiv trace grew past the floor; this test no longer exercises single-boundary skips"
        );

        let summary = backend
            .execute(&program, inputs, CHUNK_SIZE)
            .expect("execute failed");
        assert_eq!(summary.trace_len, eager_rows.len());

        let first = backend
            .replay_chunk(&summary.checkpoints[0])
            .expect("replay failed")
            .rows()
            .to_vec();
        assert_eq!(first.as_slice(), &eager_rows[..CHUNK_SIZE]);

        let last_mark = (summary.checkpoints.len() - 1) * CHUNK_SIZE;
        let last = backend
            .replay_chunk(summary.checkpoints.last().expect("nonempty trace"))
            .expect("replay failed")
            .rows()
            .to_vec();
        assert_eq!(last.as_slice(), &eager_rows[last_mark..]);
    }

    /// Advice-tape plumbing: a seeded tape reaches the emulator and the
    /// populated tape is captured on output, for both the eager and the
    /// chunked path (muldiv never consumes the tape, so it round-trips
    /// unchanged).
    #[test]
    fn advice_tape_seeds_and_captures() {
        let (program, inputs) = muldiv_setup();
        let seeded = vec![1u8, 2, 3, 4, 5];
        let inputs = inputs.with_advice_tape(Some(seeded.clone()));

        let mut backend = TracerBackend::new();
        let output = backend
            .trace(&program, inputs.clone())
            .expect("eager trace failed");
        assert_eq!(output.advice_tape, Some(seeded.clone()));

        let summary = backend
            .execute(&program, inputs, 100)
            .expect("execute failed");
        assert_eq!(summary.advice_tape, Some(seeded));
    }
}

#[cfg(test)]
#[cfg_attr(feature = "field-inline", expect(clippy::unwrap_used))]
mod tests {
    use super::*;
    use crate::emulator::elf_analyzer::test_elf::{build_elf64, StrtabOrder};
    use common::jolt_device::MemoryConfig;
    use jolt_program::execution::{TraceError, TraceInputs};

    #[cfg(feature = "field-inline")]
    use crate::{
        emulator::{cpu::Cpu, default_terminal::DefaultTerminal},
        instruction::Instruction,
    };

    #[test]
    #[expect(clippy::expect_used, reason = "test-only assertions")]
    fn tracer_backend_traces_a_guest_elf_into_jolt_rows() {
        // addi x1, x0, 1 ; addi x2, x1, 2 ; j .
        let elf = build_elf64(
            &[0x0010_0093, 0x0020_8113, 0x0000_006f],
            &[],
            StrtabOrder::GnuLd,
        );
        let program = jolt_program::build_jolt_program(&elf).expect("prepare program");
        let inputs = TraceInputs {
            memory_config: MemoryConfig {
                program_size: Some(elf.len() as u64),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut backend = TracerBackend::new();
        let output = backend.trace(&program, inputs).expect("trace succeeds");

        let rows = output.trace.rows();
        assert!(rows.len() >= 3, "two ADDIs plus the jump expansion");
        let rd = rows[0].registers().rd.expect("first ADDI writes rd");
        assert_eq!((rd.register, rd.pre_value, rd.post_value), (1, 0, 1));
        let rs1 = rows[1].registers().rs1.expect("second ADDI reads rs1");
        assert_eq!((rs1.register, rs1.value), (1, 1));
        let rd = rows[1].registers().rd.expect("second ADDI writes rd");
        assert_eq!((rd.register, rd.pre_value, rd.post_value), (2, 0, 3));

        let image = output.final_memory.expect("memory image present");
        assert!(!image.bytes.is_empty());
        assert!(!output.device.panic);
    }

    #[test]
    fn tracer_backend_rejects_programs_without_elf_bytes() {
        let program = JoltProgram::from_elf_bytes(Vec::new());
        let mut backend = TracerBackend::with_elf_path(PathBuf::from("/nonexistent"));
        assert!(matches!(
            backend.trace(&program, TraceInputs::default()),
            Err(TraceError::MissingElfBytes)
        ));
    }
    #[cfg(feature = "field-inline")]
    use jolt_riscv::{FieldInlineOp, FIELD_INLINE_OPCODE};

    #[cfg(feature = "field-inline")]
    pub(super) fn field_inline_word(op: FieldInlineOp, rd: u8, rs1: u8, rs2_or_imm: u16) -> u32 {
        let base =
            u32::from(FIELD_INLINE_OPCODE) | (u32::from(rd) << 7) | (u32::from(op.funct3()) << 12);
        match op.funct7() {
            Some(funct7) => {
                base | (u32::from(rs1) << 15)
                    | (u32::from(rs2_or_imm & 0x1f) << 20)
                    | (u32::from(funct7) << 25)
            }
            None => base | (u32::from(rs2_or_imm & 0x0fff) << 20),
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn parallel_collection_preserves_sparse_field_event_positions() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.write_register(5, 11);
        let instruction = Instruction::decode(
            field_inline_word(FieldInlineOp::LoadAccumulateFromRegister, 2, 5, 0),
            0x8000_0000,
            false,
        )
        .unwrap();
        let bytecode =
            BytecodePCMapper::try_new(&[instruction.try_jolt_instruction_row().unwrap()]).unwrap();
        let positions = [
            0,
            PARALLEL_ROW_CONVERSION_THRESHOLD - 1,
            PARALLEL_ROW_CONVERSION_THRESHOLD,
            2 * PARALLEL_ROW_CONVERSION_THRESHOLD,
        ];
        let mut trace = vec![Cycle::NoOp; 2 * PARALLEL_ROW_CONVERSION_THRESHOLD + 1];
        for position in positions {
            let mut captured = Vec::new();
            instruction.trace(&mut cpu, Some(&mut captured));
            trace[position] = captured[0];
        }
        let (data, _) = collect_rows(&trace, &bytecode).unwrap();
        assert_eq!(data.len(), trace.len());
        let event_cycles: Vec<_> = data
            .field_events()
            .iter()
            .map(|event| event.cycle)
            .collect();
        assert_eq!(event_cycles, positions);
        for (cycle, (row, captured)) in data.rows().iter().zip(&trace).enumerate() {
            if let Some(payload) = data.field_inline(cycle) {
                assert_eq!(*payload, captured.field_inline_trace().unwrap());
                assert_eq!(row.pc(), 1);
            } else {
                assert_eq!(*row, JoltTraceRow::default());
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod conversion_tests {
    use super::*;
    use crate::instruction::{ld::LD, sd::SD, RISCVCycle};
    use jolt_riscv::{RamRead, RamWrite};

    #[test]
    fn conversion_preserves_rows_and_reduces_bounds() {
        for length in [8, PARALLEL_ROW_CONVERSION_THRESHOLD + 1] {
            for (read, write) in [
                (0, 0),
                (0, u64::MAX),
                (u64::MAX, 0),
                (0x8000_2000, 0x8000_1000),
            ] {
                let mut cycles = vec![Cycle::NoOp; length];
                cycles[1] = Cycle::LD(RISCVCycle {
                    instruction: LD {
                        address: 0x8000_0000,
                        ..Default::default()
                    },
                    ram_access: RamRead {
                        address: read,
                        value: 0,
                    },
                    ..Default::default()
                });
                cycles[2] = Cycle::SD(RISCVCycle {
                    instruction: SD {
                        address: 0x8000_0004,
                        ..Default::default()
                    },
                    ram_access: RamWrite {
                        address: write,
                        pre_value: 0,
                        post_value: 0,
                    },
                    ..Default::default()
                });
                let instructions = cycles[1..3]
                    .iter()
                    .map(|cycle| cycle.instruction().try_jolt_instruction_row().unwrap())
                    .collect::<Vec<_>>();
                let bytecode = BytecodePCMapper::try_new(&instructions).unwrap();
                let (data, dimensions) = collect_rows(&cycles, &bytecode).unwrap();
                assert_eq!(data.len(), length);
                assert_eq!(data.rows()[1].ram_address(), read);
                assert_eq!(data.rows()[2].ram_address(), write);
                assert!(data.rows()[3..]
                    .iter()
                    .all(|row| *row == JoltTraceRow::default()));
                assert_eq!(dimensions.trace_length, 3);
                assert!(!dimensions.ends_in_jump);
                assert_eq!(
                    dimensions.ram_bounds.min().map(|address| address.get()),
                    [read, write]
                        .into_iter()
                        .filter(|address| *address != 0)
                        .min()
                );
                assert_eq!(
                    dimensions.ram_bounds.max().map(|address| address.get()),
                    [read, write]
                        .into_iter()
                        .filter(|address| *address != 0)
                        .max()
                );
                let empty_bytecode = BytecodePCMapper::try_new(&[]).unwrap();
                assert!(matches!(
                    collect_rows(&cycles, &empty_bytecode),
                    Err(TraceError::MissingBytecodePc { .. })
                ));
            }
        }
    }
}

#[cfg(test)]
mod limited_tests;
