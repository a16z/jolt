use std::mem::MaybeUninit;
use std::{path::PathBuf, sync::Arc};

use jolt_program::execution::{
    ChunkedExecutionBackend, DimensionFold, ExecutionBackend, ExecutionDimensions,
    ExecutionSummary, JoltProgram, MemoryImage, RamAddressBounds, RowReader, RowReplay, TraceData,
    TraceError, TraceInputs, TraceOutput,
};
use jolt_program::preprocess::BytecodePCMapper;
#[cfg(feature = "field-inline")]
use jolt_program::{execution::FieldEvent, field_inline::FieldInlineTraceData};
use jolt_riscv::{CircuitFlags, JoltTraceRow};
use rayon::prelude::*;

use common::jolt_device::JoltDevice;

use crate::emulator::cpu::AdviceTape;
use crate::emulator::decode_cache::DecodeCache;
use crate::instruction::Cycle;
#[cfg(feature = "field-inline")]
use crate::instruction::RISCVCycle;
use crate::parallel::{ChunkCheckpoint, ChunkWorker, PassEnd, PassOne};
use crate::trace_row::cycle_to_trace_row;

#[derive(Default, Debug, Clone)]
pub struct TracerBackend {
    pub elf_path: Option<PathBuf>,
}

impl TracerBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_elf_path(elf_path: PathBuf) -> Self {
        Self {
            elf_path: Some(elf_path),
        }
    }

    /// [`ExecutionBackend::trace`] with an explicit serial batch size. The
    /// serial path appends its pending cycles to the resident rows after every
    /// tick that leaves at least `batch_rows` pending, and at the end, so it
    /// buffers fewer than `batch_rows` plus one tick's cycles; a conversion
    /// error ends execution at its batch. The two-pass path (`TRACER_PARALLEL`)
    /// still converts its whole cycle vector.
    #[tracing::instrument(skip_all, name = "trace")]
    fn trace_batched(
        &self,
        program: &JoltProgram,
        inputs: TraceInputs,
        batch_rows: usize,
    ) -> Result<TraceOutput, TraceError> {
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
        let (outputs, (data, dimensions)) = match crate::parallel_config_from_env() {
            Some(config) => {
                let (cycles, emulator) = crate::parallel::run_two_pass(emulator, &config);
                let outputs = crate::finish_emulator(emulator);
                (outputs, collect_rows(&cycles, &bytecode)?)
            }
            None => {
                let mut rows = Vec::with_capacity(crate::trace_capacity_reserve());
                let mut chunks = Vec::new();
                let mut cycles = Vec::with_capacity(batch_rows);
                let mut prev_pc = 0;
                loop {
                    let pending = cycles.len();
                    crate::step_emulator(&mut emulator, &mut prev_pc, Some(&mut cycles));
                    // As in `crate::trace`, the first step that emits no row ends the run.
                    if cycles.len() == pending {
                        break;
                    }
                    if cycles.len() >= batch_rows {
                        chunks.extend(convert_rows(&mut rows, &cycles, &bytecode)?);
                        cycles.clear();
                    }
                }
                let outputs = crate::finish_emulator(emulator);
                chunks.extend(convert_rows(&mut rows, &cycles, &bytecode)?);
                (outputs, finish_rows(rows, chunks)?)
            }
        };
        let (advice_tape, final_memory, device) = outputs;
        Ok(TraceOutput::with_dimensions(
            dimensions,
            Arc::new(data),
            device,
            Some(MemoryImage {
                bytes: final_memory.materialized_nonzero_bytes(),
            }),
            Some(advice_tape.into_bytes()),
        ))
    }
}

impl ExecutionBackend for TracerBackend {
    type Trace = Arc<TraceData>;

    fn trace(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput, TraceError> {
        self.trace_batched(program, inputs, SERIAL_BATCH_ROWS)
    }
}

const PARALLEL_ROW_CONVERSION_THRESHOLD: usize = 1 << 14;

/// Pending cycles at which the serial trace converts a batch: 96 MiB of
/// default-build cycles, 64 parallel conversion chunks.
const SERIAL_BATCH_ROWS: usize = 1 << 20;

struct ConvertedChunk {
    bounds: RamAddressBounds,
    #[cfg(feature = "field-inline")]
    fields: Vec<FieldEvent>,
}

fn collect_rows(
    cycles: &[Cycle],
    bytecode: &BytecodePCMapper,
) -> Result<(TraceData, ExecutionDimensions), TraceError> {
    let mut rows = Vec::with_capacity(cycles.len());
    let chunks = convert_rows(&mut rows, cycles, bytecode)?;
    finish_rows(rows, chunks)
}

/// Appends the rows of `cycles`, which follow the cycles of every row already
/// in `rows`; field-inline events carry their index in `rows`.
fn convert_rows(
    rows: &mut Vec<JoltTraceRow>,
    cycles: &[Cycle],
    bytecode: &BytecodePCMapper,
) -> Result<Vec<ConvertedChunk>, TraceError> {
    let parallel = cycles.len() > PARALLEL_ROW_CONVERSION_THRESHOLD;
    let _span = tracing::info_span!(
        "trace_rows_from_cycles",
        cycles = cycles.len(),
        mode = if parallel { "parallel" } else { "serial" }
    )
    .entered();
    let start = rows.len();
    rows.reserve(cycles.len());
    let slots = &mut rows.spare_capacity_mut()[..cycles.len()];
    let fill = |first: usize, slots: &mut [MaybeUninit<JoltTraceRow>], cycles: &[Cycle]| {
        #[cfg(not(feature = "field-inline"))]
        let _ = first;
        let mut chunk = ConvertedChunk {
            bounds: RamAddressBounds::default(),
            #[cfg(feature = "field-inline")]
            fields: Vec::new(),
        };
        for (offset, (slot, cycle)) in slots.iter_mut().zip(cycles).enumerate() {
            #[cfg(not(feature = "field-inline"))]
            let _ = offset;
            let row = cycle_to_trace_row(cycle, bytecode)?;
            chunk.bounds.observe(row.ram_address());
            slot.write(row);
            #[cfg(feature = "field-inline")]
            if let Some(data) = cycle.field_inline_trace() {
                chunk.fields.push(FieldEvent {
                    cycle: first + offset,
                    data,
                });
            }
        }
        Ok::<_, TraceError>(chunk)
    };
    let chunks = if parallel {
        slots
            .par_chunks_mut(PARALLEL_ROW_CONVERSION_THRESHOLD)
            .zip(cycles.par_chunks(PARALLEL_ROW_CONVERSION_THRESHOLD))
            .enumerate()
            .map(|(index, (slots, cycles))| {
                fill(
                    start + index * PARALLEL_ROW_CONVERSION_THRESHOLD,
                    slots,
                    cycles,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![fill(start, slots, cycles)?]
    };
    // SAFETY: successful workers initialized the `cycles.len()` slots from
    // `start`. JoltTraceRow is Copy, so partially written rows need no cleanup
    // on error or unwinding.
    unsafe { rows.set_len(start + cycles.len()) };
    Ok(chunks)
}

/// `chunks` are the results of every [`convert_rows`] call that filled `rows`,
/// in call order.
fn finish_rows(
    rows: Vec<JoltTraceRow>,
    chunks: Vec<ConvertedChunk>,
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
        let mut fields = Vec::with_capacity(chunks.iter().map(|chunk| chunk.fields.len()).sum());
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

/// Production checkpoint spacing for [`TracerBackend::execute_replay`].
pub const REPLAY_SEGMENT_ROWS: usize = 1 << 15;

/// Rows each census task converts per read; bounds its row buffer.
const CENSUS_BATCH_ROWS: usize = 1 << 12;

/// Smallest boundary spacing of the trait's [`ChunkedExecutionBackend::execute`]:
/// each boundary holds a CPU state, including its 32 KiB CSR file, and the
/// touched memory prefix, so a boundary per tiny chunk would grow with the
/// chunk count. A chunk replays from the last boundary at or before its first
/// row, so its cost stays below spacing + chunk size + one tick's rows.
const MIN_BOUNDARY_SPACING_ROWS: usize = 1 << 16;

/// Static parts every replay worker starts from, taken before pass-1 runs.
struct WorkerSeed {
    device: Option<JoltDevice>,
    decode: DecodeCache,
    bytecode: BytecodePCMapper,
}

/// Pass-1 state at a tick boundary and the touched prefix of guest memory
/// there (every later word is zero).
struct ReplayBoundary {
    rows: usize,
    state: ChunkCheckpoint,
    memory: Box<[u64]>,
}

impl ReplayBoundary {
    fn capture(pass: &PassOne) -> Self {
        let (image, touched) = pass
            .emulator()
            .get_cpu()
            .mmu
            .memory
            .memory
            .data
            .flat_parts();
        Self {
            rows: pass.rows(),
            state: pass.checkpoint(),
            memory: image[..touched].into(),
        }
    }
}

/// Pass-1 boundaries from which any range of the `executed_rows` rows can be
/// re-executed.
struct Recording {
    /// Ascending by `rows`, starting at row 0: the first tick boundary at or
    /// after each multiple of `spacing` (multiples inside one tick share it).
    boundaries: Vec<ReplayBoundary>,
    /// State after the last tick that emitted rows, at `executed_rows`.
    final_state: ChunkCheckpoint,
    executed_rows: usize,
    spacing: usize,
    /// Pass-1's final touched memory prefix. A replay repeats pass-1's ticks
    /// from identical state, so it accesses no word past it; a diverged
    /// replay that does hits the flat backing's out-of-bounds panic.
    resident_words: usize,
    /// Guest memory size reported for address validation.
    logical_words: usize,
    seed: WorkerSeed,
}

struct RecordedOutputs {
    device: JoltDevice,
    final_memory: MemoryImage,
    advice_tape: Vec<u8>,
}

impl Recording {
    fn worker(&self, boundary: &ReplayBoundary) -> ChunkWorker {
        let mut worker = ChunkWorker::from_seed(self.seed.device.clone(), self.seed.decode.clone());
        worker.install_prefix(
            &boundary.state,
            &boundary.memory,
            self.resident_words,
            self.logical_words,
        );
        worker
    }

    /// The `index`-th boundary a forward replay checks: the stored
    /// boundaries, then the final state.
    fn boundary_state(&self, index: usize) -> Option<(usize, &ChunkCheckpoint)> {
        match self.boundaries.get(index) {
            Some(boundary) => Some((boundary.rows, &boundary.state)),
            None => {
                (index == self.boundaries.len()).then_some((self.executed_rows, &self.final_state))
            }
        }
    }

    /// State at `executed_rows` when pass-1 ended on a tick that emitted no
    /// row and so already advanced past it: re-execute the last segment on
    /// pass-1's execute path.
    #[expect(clippy::expect_used)]
    fn final_row_state(&self) -> Result<ChunkCheckpoint, TraceError> {
        let last = self.boundaries.last().expect("row 0 boundary recorded");
        let mut worker = self.worker(last);
        while worker.cpu().trace_len < self.executed_rows {
            let rows = worker.cpu().trace_len;
            worker.execute_tick();
            if worker.cpu().trace_len == rows {
                return Err(TraceError::ReplayDiverged {
                    row: rows,
                    reason: "a re-executed tick emitted no row".to_string(),
                });
            }
        }
        if worker.cpu().trace_len != self.executed_rows {
            return Err(TraceError::ReplayDiverged {
                row: self.executed_rows,
                reason: format!(
                    "re-execution ended at row {}, past the last row",
                    worker.cpu().trace_len
                ),
            });
        }
        Ok(ChunkCheckpoint::capture(worker.cpu()))
    }

    /// Execution dimensions of every executed row, read by one cursor per
    /// segment in parallel. Each cursor checks the boundaries its ticks reach,
    /// so every boundary and the final state are compared against replay.
    fn census(&self) -> Result<ExecutionDimensions, TraceError> {
        let segments = self.executed_rows.div_ceil(self.spacing);
        let _span = tracing::info_span!("TracerReplay::census", segments).entered();
        let fold = (0..segments)
            .into_par_iter()
            .map(|segment| {
                let start = segment * self.spacing;
                let end = (start + self.spacing).min(self.executed_rows);
                let mut cursor = ReplayCursor::open(self, start, end)?;
                let mut rows = vec![JoltTraceRow::default(); (end - start).min(CENSUS_BATCH_ROWS)];
                let mut fold = DimensionFold::default();
                for first in (start..end).step_by(rows.len()) {
                    let batch = &mut rows[..(end - first).min(CENSUS_BATCH_ROWS)];
                    cursor.read(batch)?;
                    fold.observe(first, batch);
                }
                Ok::<_, TraceError>(fold)
            })
            .try_reduce(DimensionFold::default, |left, right| Ok(left.merge(right)))?;
        Ok(fold.dimensions())
    }

    fn log_storage(&self) {
        let rows = self.boundaries.iter().map(|boundary| boundary.rows);
        let max_gap = rows
            .clone()
            .skip(1)
            .chain([self.executed_rows])
            .zip(rows.clone())
            .map(|(next, previous)| next - previous)
            .max();
        tracing::info!(
            boundaries = self.boundaries.len(),
            boundary_index_bytes = self.boundaries.capacity() * size_of::<ReplayBoundary>(),
            state_bytes = self
                .boundaries
                .iter()
                .map(|boundary| boundary.state.heap_bytes())
                .sum::<usize>()
                + self.final_state.heap_bytes(),
            prefix_bytes = self
                .boundaries
                .iter()
                .map(|boundary| size_of_val(&*boundary.memory))
                .sum::<usize>(),
            max_prefix_words = self
                .boundaries
                .iter()
                .map(|boundary| boundary.memory.len())
                .max(),
            max_gap,
            max_mark_offset = rows.map(|rows| rows % self.spacing).max(),
            executed_rows = self.executed_rows,
            resident_words = self.resident_words,
            logical_words = self.logical_words,
            "replay recording"
        );
    }
}

/// Sequential cursor over a recording. Re-executes from the last boundary at
/// or before its first row, keeps the rows of the current tick so a seek or
/// read may end inside a multi-row tick, and compares the replay state at
/// every later boundary a tick reaches; a tick that passes a boundary without
/// ending on it is a divergence. Comparisons cover the CPU and device state
/// of [`ChunkCheckpoint::diff_vs_cpu`], not memory.
struct ReplayCursor<'a> {
    recording: &'a Recording,
    worker: ChunkWorker,
    /// Index of the next [`Recording::boundary_state`] to compare.
    next_check: usize,
    /// Rows of the latest tick; `tick[delivered..]` are still pending.
    tick: Vec<Cycle>,
    delivered: usize,
    /// First row the cursor may not deliver.
    limit: usize,
}

impl<'a> ReplayCursor<'a> {
    fn open(recording: &'a Recording, start: usize, limit: usize) -> Result<Self, TraceError> {
        if start >= limit || limit > recording.executed_rows {
            return Err(TraceError::ReplayOutOfRange {
                start,
                end: start.saturating_add(1),
                extent: limit.min(recording.executed_rows),
            });
        }
        let index = recording
            .boundaries
            .partition_point(|boundary| boundary.rows <= start)
            - 1;
        let boundary = &recording.boundaries[index];
        let mut cursor = Self {
            recording,
            worker: recording.worker(boundary),
            next_check: index + 1,
            tick: Vec::new(),
            delivered: 0,
            limit,
        };
        let mut skip = start - boundary.rows;
        while skip > 0 {
            if cursor.delivered == cursor.tick.len() {
                cursor.run_tick()?;
            }
            let rows = skip.min(cursor.tick.len() - cursor.delivered);
            cursor.delivered += rows;
            skip -= rows;
        }
        Ok(cursor)
    }

    /// Index of the next row to deliver.
    fn position(&self) -> usize {
        self.worker.cpu().trace_len - (self.tick.len() - self.delivered)
    }

    fn claim(&self, rows: usize) -> Result<(), TraceError> {
        let start = self.position();
        let end = start.saturating_add(rows);
        if end > self.limit {
            return Err(TraceError::ReplayOutOfRange {
                start,
                end,
                extent: self.limit,
            });
        }
        Ok(())
    }

    fn next_cycle(&mut self) -> Result<&Cycle, TraceError> {
        if self.delivered == self.tick.len() {
            self.run_tick()?;
        }
        let cycle = &self.tick[self.delivered];
        self.delivered += 1;
        Ok(cycle)
    }

    fn run_tick(&mut self) -> Result<(), TraceError> {
        self.tick.clear();
        self.delivered = 0;
        self.worker.run_ticks(1, &mut self.tick);
        let rows = self.worker.cpu().trace_len;
        if self.tick.is_empty() {
            return Err(TraceError::ReplayDiverged {
                row: rows,
                reason: "a replayed tick emitted no row".to_string(),
            });
        }
        while let Some((at, state)) = self.recording.boundary_state(self.next_check) {
            if at > rows {
                break;
            }
            let mismatch = if at < rows {
                Some(format!("a tick ended at row {rows}, past the boundary"))
            } else {
                state.diff_vs_cpu(self.worker.cpu())
            };
            if let Some(reason) = mismatch {
                return Err(TraceError::ReplayDiverged { row: at, reason });
            }
            self.next_check += 1;
        }
        Ok(())
    }
}

impl RowReader for ReplayCursor<'_> {
    fn read(&mut self, rows: &mut [JoltTraceRow]) -> Result<(), TraceError> {
        self.claim(rows.len())?;
        let recording = self.recording;
        for row in rows {
            *row = cycle_to_trace_row(self.next_cycle()?, &recording.seed.bytecode)?;
        }
        Ok(())
    }
}

/// An execution recorded by [`TracerBackend::execute_replay`], whose rows are
/// re-executed on demand through [`RowReplay`] readers. It retains one
/// boundary per [`RowReplay::segment_rows`] mark with the touched memory
/// prefix, never rows.
pub struct TracerReplay {
    recording: Recording,
    proof_len: usize,
}

impl RowReplay for TracerReplay {
    fn segment_rows(&self) -> usize {
        self.recording.spacing
    }

    fn proof_len(&self) -> usize {
        self.proof_len
    }

    fn reader(&self, start: usize) -> Result<Box<dyn RowReader + '_>, TraceError> {
        Ok(Box::new(ReplayCursor::open(
            &self.recording,
            start,
            self.proof_len,
        )?))
    }
}

/// A resume point for the chunked-execution contract: one chunk of a shared
/// recording, replayed by the same cursor as [`TracerReplay`] readers.
pub struct TracerChunkCheckpoint {
    recording: Arc<Recording>,
    first_row: usize,
    rows: usize,
}

impl TracerBackend {
    /// Execute without retaining rows: pass-1 records a boundary per
    /// `segment_rows` mark (a power of two), then a parallel census replays
    /// every row once to derive the execution dimensions, comparing replay
    /// against every boundary. Rows are re-executed on demand through the
    /// returned [`TracerReplay`].
    pub fn execute_replay(
        &self,
        program: &JoltProgram,
        inputs: TraceInputs,
        segment_rows: usize,
    ) -> Result<TraceOutput<TracerReplay>, TraceError> {
        #[cfg(feature = "field-inline")]
        if program.profile.supports_field_inline() {
            return Err(TraceError::ReplayUnsupported(
                "field-inline instruction profiles",
            ));
        }
        if !segment_rows.is_power_of_two() {
            return Err(TraceError::Backend("segment_rows must be a power of two"));
        }
        let (recording, outputs) = self.record(program, inputs, segment_rows)?;
        let dimensions = recording.census()?;
        Ok(TraceOutput::with_dimensions(
            dimensions,
            TracerReplay {
                recording,
                proof_len: dimensions.trace_length,
            },
            outputs.device,
            Some(outputs.final_memory),
            Some(outputs.advice_tape),
        ))
    }

    /// Pass-1: execute without rows, keeping a boundary at the first tick
    /// boundary at or after every multiple of `spacing` rows (nonzero).
    fn record(
        &self,
        program: &JoltProgram,
        inputs: TraceInputs,
        spacing: usize,
    ) -> Result<(Recording, RecordedOutputs), TraceError> {
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        let _span = tracing::info_span!("TracerReplay::record", spacing).entered();
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
        let seed = WorkerSeed {
            device: emulator.get_cpu().mmu.jolt_device.clone(),
            decode: emulator
                .get_cpu()
                .mmu
                .decode_cache
                .snapshot_with_empty_entries(),
            bytecode,
        };

        let mut pass = PassOne::new(emulator);
        let mut boundaries = vec![ReplayBoundary::capture(&pass)];
        let mut next_mark = spacing;
        while pass.step() {
            if pass.rows() >= next_mark {
                boundaries.push(ReplayBoundary::capture(&pass));
                next_mark = (pass.rows() / spacing + 1) * spacing;
            }
        }
        boundaries.shrink_to_fit();
        let memory = &pass.emulator().get_cpu().mmu.memory.memory.data;
        let mut recording = Recording {
            boundaries,
            final_state: pass.checkpoint(),
            executed_rows: pass.rows(),
            spacing,
            resident_words: memory.flat_parts().1,
            logical_words: memory.get_num_doublewords(),
            seed,
        };
        if pass.end() == Some(PassEnd::ZeroRowTick) {
            recording.final_state = recording.final_row_state()?;
        }
        recording.log_storage();

        let (advice_tape, final_memory, device) = crate::finish_emulator(pass.into_emulator());
        Ok((
            recording,
            RecordedOutputs {
                device,
                final_memory: MemoryImage {
                    bytes: final_memory.materialized_nonzero_bytes(),
                },
                advice_tape: advice_tape.into_bytes(),
            },
        ))
    }

    /// [`ChunkedExecutionBackend::execute`] with an explicit boundary spacing.
    /// The trait method uses at least [`MIN_BOUNDARY_SPACING_ROWS`]; tests pass
    /// the chunk size to exercise multi-boundary selection on guests whose
    /// whole trace is shorter than that floor.
    fn execute_chunked(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
        chunk_size: usize,
        boundary_spacing: usize,
    ) -> Result<ExecutionSummary<TracerChunkCheckpoint>, TraceError> {
        if chunk_size == 0 {
            return Err(TraceError::Backend("chunk_size must be nonzero"));
        }
        let (recording, outputs) = self.record(program, inputs, boundary_spacing)?;
        let trace_len = recording.executed_rows;
        let recording = Arc::new(recording);
        let checkpoints = (0..trace_len.div_ceil(chunk_size))
            .map(|chunk| {
                let first_row = chunk * chunk_size;
                TracerChunkCheckpoint {
                    recording: Arc::clone(&recording),
                    first_row,
                    rows: chunk_size.min(trace_len - first_row),
                }
            })
            .collect();
        Ok(ExecutionSummary {
            checkpoints,
            trace_len,
            device: outputs.device,
            final_memory: Some(outputs.final_memory),
            advice_tape: Some(outputs.advice_tape),
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
        self.execute_chunked(
            program,
            inputs,
            chunk_size,
            chunk_size.max(MIN_BOUNDARY_SPACING_ROWS),
        )
    }

    fn replay_chunk(&self, checkpoint: &Self::Checkpoint) -> Result<TraceData, TraceError> {
        let mut cursor = ReplayCursor::open(
            &checkpoint.recording,
            checkpoint.first_row,
            checkpoint.first_row + checkpoint.rows,
        )?;
        let mut cycles = Vec::with_capacity(checkpoint.rows);
        for _ in 0..checkpoint.rows {
            cycles.push(*cursor.next_cycle()?);
        }
        collect_rows(&cycles, &checkpoint.recording.seed.bytecode).map(|(data, _)| data)
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

    /// Serial batches end at tick boundaries once full. `last` is the final
    /// row that a multi-row tick continues past, so batches of `last + 1` rows
    /// overrun into the rest of that tick and leave the remaining rows to one
    /// partial final batch. Rows, dimensions and outputs equal one conversion
    /// of `crate::trace`'s whole cycle vector.
    #[test]
    fn tracer_backend_batches_match_whole_trace_conversion() {
        let (program, inputs) = muldiv_setup();
        let (_, cycles, final_memory, device, advice_tape) = crate::trace(
            program.elf_bytes(),
            None,
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            None,
        );
        let bytecode = BytecodePCMapper::try_new(&program.expanded_bytecode).expect("bytecode");
        let (whole, dimensions) = collect_rows(&cycles, &bytecode).expect("whole conversion");
        let rows = whole.rows();
        let last = rows
            .iter()
            .rposition(|row| {
                row.virtual_sequence_remaining()
                    .is_some_and(|left| left > 0)
            })
            .expect("a multi-row tick");
        assert!(
            (1..=last).contains(&(rows.len() - last - 2)),
            "muldiv no longer ends in a short tail after its last multi-row tick"
        );

        let batched = TracerBackend::new()
            .trace_batched(&program, inputs, last + 1)
            .expect("batched trace");
        assert_eq!(batched.trace.rows(), rows);
        assert_eq!(batched.dimensions, dimensions);
        assert_eq!(batched.device, device);
        assert_eq!(
            batched.final_memory,
            Some(MemoryImage {
                bytes: final_memory.materialized_nonzero_bytes()
            })
        );
        assert_eq!(batched.advice_tape, Some(advice_tape.into_bytes()));
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test module")]
mod replay_tests {
    use super::*;
    use crate::emulator::elf_analyzer::test_elf::{build_elf64, StrtabOrder};
    use crate::test_utils::{build_fibonacci_guest, build_muldiv_guest};
    use common::jolt_device::MemoryConfig;
    use jolt_program::execution::build_jolt_program;

    fn setup(elf: &[u8], input: Vec<u8>) -> (JoltProgram, TraceInputs) {
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let program = build_jolt_program(elf).expect("failed to build Jolt program");
        let inputs = TraceInputs::new(input, Vec::new(), Vec::new(), memory_config);
        (program, inputs)
    }

    fn muldiv() -> (JoltProgram, TraceInputs) {
        setup(
            &build_muldiv_guest(),
            vec![0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c],
        )
    }

    /// Reads `rows` rows from `start` in batches of 2, 3, 5, 9, ... rows.
    fn read_rows(
        trace: &TracerReplay,
        start: usize,
        rows: usize,
    ) -> Result<Vec<JoltTraceRow>, TraceError> {
        let mut reader = trace.reader(start)?;
        let mut read = Vec::with_capacity(rows);
        for batch in (0..).map(|power| (1usize << power) + 1) {
            let mut buffer = vec![JoltTraceRow::default(); batch.min(rows - read.len())];
            reader.read(&mut buffer)?;
            read.extend(buffer);
            if read.len() == rows {
                break;
            }
        }
        assert!(matches!(
            reader.read(&mut [JoltTraceRow::default()]),
            Err(TraceError::ReplayOutOfRange { .. })
        ));
        Ok(read)
    }

    /// Readers opened at every start, in reverse order, deliver the eager rows
    /// through the end of the proof prefix and no further, across segment
    /// sizes from one row to beyond the trace; the census reproduces the eager
    /// dimensions, device, final memory and advice tape.
    #[test]
    fn replay_readers_match_eager_rows_from_every_start() {
        let fibonacci = setup(
            &build_fibonacci_guest(),
            postcard::to_stdvec(&50u32).expect("serialize input"),
        );
        for (name, (program, inputs)) in [("muldiv", muldiv()), ("fibonacci", fibonacci)] {
            let eager = TracerBackend::new()
                .trace(&program, inputs.clone())
                .expect("eager trace");
            let rows = eager.trace.proof_rows();
            assert_eq!(rows.len(), eager.trace.len(), "{name}: no padding rows");
            assert!(
                rows.iter()
                    .any(|row| row.virtual_sequence_remaining().is_some()
                        && !row.circuit_flags().get(CircuitFlags::IsFirstInSequence)),
                "{name}: starts include virtual-sequence interiors"
            );
            for segment_rows in [1, 2, 64, 1 << 10, 1 << 20] {
                let case = format!("{name}, segment_rows {segment_rows}");
                let replay = TracerBackend::new()
                    .execute_replay(&program, inputs.clone(), segment_rows)
                    .expect("replay");
                assert_eq!(replay.dimensions, eager.dimensions, "{case}");
                assert_eq!(replay.device, eager.device, "{case}");
                assert_eq!(replay.final_memory, eager.final_memory, "{case}");
                assert_eq!(replay.advice_tape, eager.advice_tape, "{case}");
                let trace = &replay.trace;
                assert_eq!(
                    (trace.segment_rows(), trace.proof_len()),
                    (segment_rows, rows.len()),
                    "{case}"
                );
                for start in (0..rows.len()).rev() {
                    let read = read_rows(trace, start, rows.len() - start).expect("read");
                    assert_eq!(read, rows[start..], "{case}, start {start}");
                }
                assert!(matches!(
                    trace.reader(rows.len()),
                    Err(TraceError::ReplayOutOfRange { .. })
                ));
                assert!(matches!(
                    ReplayCursor::open(&trace.recording, 0, rows.len() + 1),
                    Err(TraceError::ReplayOutOfRange { .. })
                ));
            }
            assert!(matches!(
                TracerBackend::new().execute_replay(&program, inputs, 3),
                Err(TraceError::Backend(_))
            ));
        }
    }

    /// A replay that disagrees with a recorded boundary fails at that
    /// boundary's row: a mismatched state, and a tick that passes the boundary
    /// row without ending on it. Each case corrupts its own recording.
    #[test]
    fn replay_rejects_a_diverging_boundary() {
        let (program, inputs) = muldiv();
        let replay = |inputs: &TraceInputs| {
            TracerBackend::new()
                .execute_replay(&program, inputs.clone(), 1)
                .expect("replay")
                .trace
        };

        let mut trace = replay(&inputs);
        let boundaries = &mut trace.recording.boundaries;
        boundaries[3].state = boundaries[2].state.clone();
        let row = boundaries[3].rows;
        let proof_len = trace.proof_len;
        assert!(matches!(
            read_rows(&trace, 0, proof_len),
            Err(TraceError::ReplayDiverged { row: at, .. }) if at == row
        ));

        let mut trace = replay(&inputs);
        let boundaries = &mut trace.recording.boundaries;
        let index = (1..boundaries.len())
            .find(|&index| boundaries[index].rows - boundaries[index - 1].rows > 1)
            .expect("a multi-row tick");
        boundaries[index].rows -= 1;
        let row = boundaries[index].rows;
        assert!(matches!(
            read_rows(&trace, 0, proof_len),
            Err(TraceError::ReplayDiverged { row: at, .. }) if at == row
        ));
    }

    /// The final boundary rebuilt by re-executing the last segment, as after
    /// a terminal zero-row tick, is the state replay reaches at the last row.
    #[test]
    fn final_row_state_matches_replay_at_the_last_row() {
        let (program, inputs) = muldiv();
        for segment_rows in [64, 1 << 20] {
            let mut trace = TracerBackend::new()
                .execute_replay(&program, inputs.clone(), segment_rows)
                .expect("replay")
                .trace;
            trace.recording.final_state = trace
                .recording
                .final_row_state()
                .expect("re-executed final state");
            let start = trace.recording.boundaries.last().expect("boundary").rows;
            read_rows(&trace, start, trace.proof_len - start).expect("final segment replay");
        }
    }

    /// `a0 = JOLT_ADVICE_WRITE_CALL_ID; a1 = pc; a2 = 8; VirtualHostIO` appends
    /// eight text bytes to the advice tape, then `ADVICE_LD x5` reads eight
    /// bytes back, crossing from the seeded tape into the appended bytes.
    fn advice_round_trip_elf() -> Vec<u8> {
        build_elf64(
            &[
                0x00ad_c537, // lui a0, 0xadc
                0xabe5_0513, // addi a0, a0, -0x542
                0x0000_0597, // auipc a1, 0
                0x0080_0613, // addi a2, x0, 8
                0x0000_205b, // VirtualHostIO
                0x0600_02db, // advice.ld x5
                0x0000_006f, // j .
            ],
            &[],
            StrtabOrder::GnuLd,
        )
    }

    /// Replay appends advice to its private restored tape: the round trip
    /// replays the eager rows whether a boundary separates the append from
    /// the read or both lie inside the final segment.
    #[test]
    fn replay_appends_advice_to_its_private_tape() {
        let elf = advice_round_trip_elf();
        let (program, inputs) = setup(&elf, Vec::new());
        let inputs = inputs.with_advice_tape(Some(vec![0xa1, 0xa2, 0xa3]));
        let eager = TracerBackend::new()
            .trace(&program, inputs.clone())
            .expect("eager trace");
        let appended: [u8; 8] = [0x97, 0x05, 0x00, 0x00, 0x13, 0x06, 0x80, 0x00];
        assert_eq!(
            eager.advice_tape.as_deref(),
            Some(
                [[0xa1u8, 0xa2, 0xa3].as_slice(), &appended]
                    .concat()
                    .as_slice()
            )
        );
        let rows = eager.trace.proof_rows();
        assert!(rows.iter().any(|row| {
            row.registers().rd.is_some_and(|write| {
                write.register == 5 && write.post_value == 0x1300_0005_97a3_a2a1
            })
        }));

        for segment_rows in [1, 1 << 20] {
            let replay = TracerBackend::new()
                .execute_replay(&program, inputs.clone(), segment_rows)
                .expect("replay");
            assert_eq!(replay.dimensions, eager.dimensions);
            assert_eq!(replay.advice_tape, eager.advice_tape);
            let read = read_rows(&replay.trace, 0, rows.len()).expect("read");
            assert_eq!(read, rows, "segment_rows {segment_rows}");
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn replay_rejects_field_inline_profiles() {
        use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE;

        let (mut program, inputs) = muldiv();
        program.profile = RV64IMAC_JOLT_FIELD_INLINE;
        assert!(matches!(
            TracerBackend::new().execute_replay(&program, inputs, 64),
            Err(TraceError::ReplayUnsupported(_))
        ));
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
    fn field_inline_word(op: FieldInlineOp, rd: u8, rs1: u8, rs2_or_imm: u16) -> u32 {
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

        // Appending the trace in three conversions keeps every event's trace
        // index: serial from row 0, serial from row 1, then parallel from the
        // threshold row with an event in its second chunk.
        let threshold = PARALLEL_ROW_CONVERSION_THRESHOLD;
        let mut rows = Vec::new();
        let mut chunks = Vec::new();
        for part in [&trace[..1], &trace[1..threshold], &trace[threshold..]] {
            chunks.extend(convert_rows(&mut rows, part, &bytecode).unwrap());
        }
        let (appended, _) = finish_rows(rows, chunks).unwrap();
        assert_eq!(appended.rows(), data.rows());
        let appended_cycles: Vec<_> = appended
            .field_events()
            .iter()
            .map(|event| event.cycle)
            .collect();
        assert_eq!(appended_cycles, positions);
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
