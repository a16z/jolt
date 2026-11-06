use std::mem::MaybeUninit;
use std::{path::PathBuf, sync::Arc};

use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, ExecutionDimensions, ExecutionSummary, JoltProgram,
    MemoryImage, RamAddressBounds, TraceData, TraceError, TraceInputs, TraceOutput,
};
#[cfg(test)]
use jolt_program::execution::{TraceAllocationError, TraceAllocationPhase};
use jolt_program::expand::MAX_EXPANDED_ROWS_PER_SOURCE;
use jolt_program::preprocess::BytecodePreprocessing;
#[cfg(feature = "field-inline")]
use jolt_program::{execution::FieldEvent, field_inline::FieldInlineTraceData};
use jolt_riscv::{CircuitFlags, JoltTraceRow};
use rayon::prelude::*;

use common::jolt_device::JoltDevice;
#[cfg(test)]
use common::jolt_device::MemoryLayout;

use crate::emulator::cpu::AdviceTape;
use crate::emulator::decode_cache::DecodeCache;
use crate::instruction::Cycle;
#[cfg(feature = "field-inline")]
use crate::instruction::RISCVCycle;
use crate::parallel::{ChunkCheckpoint, ChunkWorker, PassOne, SnapshotPool};
mod limited;
use crate::trace_row::cycle_to_trace_row;
#[cfg(test)]
use limited::LimitedTrace;
use limited::{LimitedSetup, TraceLimit};

#[derive(Default, Debug, Clone)]
pub struct TracerBackend {
    pub elf_path: Option<PathBuf>,
    trace_allocation_limit: Option<TraceLimit>,
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

    /// Selects limited serial tracing when the original trace's row payload
    /// over `row_domain` exceeds `limit_bytes`, or when that payload overflows.
    /// Otherwise returns `None`, preserving the original backend and its
    /// `TRACER_PARALLEL` opt-in.
    ///
    /// The original payload proxy is `D = size_of::<Cycle>() * max(R0, N) +
    /// size_of::<JoltTraceRow>() * N`, where `N = row_domain` and `R0` is
    /// `JOLT_TRACER_CAPACITY_ROWS` (default 2^24): original tracing reserves
    /// `R0` cycles; limited tracing instead reserves `min(R0, N)` rows and
    /// checks subsequent growth. The domain neither counts nor caps emitted
    /// rows, and D is not a peak: a short run can take the limited route even
    /// when the original fits, while an original trace admitted at D can
    /// exceed the limit through its uncounted requests or cycle-buffer growth.
    ///
    /// Limited tracing checks requested payload before allocation, counting
    /// buffers at observed capacity; a refusal returns a typed error without
    /// output, but does not roll back guest effects or earlier allocations.
    ///
    /// Counted:
    /// - Bytecode preprocessing, including its input copy and PC map.
    /// - Flat guest memory, the emulator's ELF copy, caller input/advice buffers,
    ///   device copies and output reserve.
    /// - Rows and their old/new buffers during growth, the pending cycle window,
    ///   converted-chunk metadata, field events and their merge.
    /// - Advice tape at each check and the final memory image.
    ///
    /// Not counted:
    /// - Advice-tape growth/write scratch between checks, ELF headers/symbols,
    ///   decode/expansion caches, inline advice and provider temporaries.
    /// - Cycle markers, prints/backtraces, the caller's program/preprocessing,
    ///   allocator/runtime overhead, and the SDK's `compute_advice_tape` trace.
    ///
    /// This is not a process-memory cap or a guarantee that proving fits.
    /// Doubling row/chunk growth can reject a near-limit request that exact
    /// growth would admit. On the limited route, the pending cycle window imposes
    /// a setup floor even for a short domain; the trace retains its row capacity for
    /// subsequent prover-budget checks.
    ///
    /// Limited tracing rejects `TRACER_PARALLEL >= 2` and
    /// [`ChunkedExecutionBackend::execute`] with
    /// [`TraceError::UnsupportedTraceAllocationMode`]; chunk replay is unchecked.
    pub fn limited_for_row_domain(limit_bytes: usize, row_domain: usize) -> Option<Self> {
        let payload = size_of::<Cycle>()
            .checked_mul(crate::trace_capacity_reserve().max(row_domain))
            .and_then(|cycles| {
                cycles.checked_add(size_of::<JoltTraceRow>().checked_mul(row_domain)?)
            });
        payload
            .is_none_or(|payload| payload > limit_bytes)
            .then(|| Self {
                trace_allocation_limit: Some(TraceLimit {
                    bytes: limit_bytes,
                    initial_rows: crate::trace_capacity_reserve().min(row_domain),
                }),
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

    /// [`ExecutionBackend::trace`] under `limit`, serially: after every tick
    /// that leaves at least `batch_rows` cycles pending, and at the end, the
    /// pending cycles are converted into the resident rows.
    #[tracing::instrument(skip_all, name = "trace")]
    fn trace_limited(
        &self,
        program: &JoltProgram,
        inputs: TraceInputs,
        limit: TraceLimit,
        batch_rows: usize,
    ) -> Result<TraceOutput, TraceError> {
        if crate::parallel_config_from_env().is_some() {
            return Err(TraceError::UnsupportedTraceAllocationMode);
        }
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        let setup = LimitedSetup::admit(limit, program, &inputs, batch_rows)?;
        let bytecode = BytecodePreprocessing::preprocess(
            program.expanded_bytecode.clone(),
            program.entry_address,
            program.profile,
        )?;
        let mut emulator = crate::create_emulator(
            program.elf_bytes(),
            self.elf_path.as_ref(),
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            inputs.advice_tape.map(AdviceTape::from_bytes),
        );
        let mut trace = setup.reserve(&mut emulator, &bytecode)?;
        loop {
            let pending = trace.cycles.len();
            crate::step_emulator(&mut emulator, Some(&mut trace.cycles));
            // As in `crate::trace`, the first step that emits no row ends the run.
            if trace.cycles.len() == pending {
                break;
            }
            if trace.cycles.len() >= batch_rows {
                let tape = emulator.get_cpu().advice_tape.capacity();
                trace.convert_pending(&bytecode, tape)?;
            }
        }
        let (advice_tape, final_memory, device) = crate::finish_emulator(emulator);
        trace.convert_pending(&bytecode, advice_tape.capacity())?;
        let (data, dimensions, image) = trace.finish(&final_memory, advice_tape.capacity())?;
        Ok(TraceOutput::with_dimensions(
            dimensions,
            Arc::new(data),
            device,
            Some(image),
            Some(advice_tape.into_bytes()),
        ))
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
        let bytecode = BytecodePreprocessing::preprocess(
            program.expanded_bytecode.clone(),
            program.entry_address,
            program.profile,
        )?;
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

/// Pending cycles at which a limited trace converts a batch: 96 MiB of
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
        bytecode: &BytecodePreprocessing,
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
    bytecode: &BytecodePreprocessing,
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
    bytecode: BytecodePreprocessing,
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

        let bytecode = BytecodePreprocessing::preprocess(
            program.expanded_bytecode.clone(),
            program.entry_address,
            program.profile,
        )?;
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

    #[test]
    fn eager_and_replay_reject_bytecode_mismatches() {
        let (mut program, inputs) = muldiv_setup();
        let mut backend = TracerBackend::new();
        let eager = backend
            .trace(&program, inputs.clone())
            .expect("eager trace failed");
        let executed = eager.trace.rows()[0].instruction();
        let instruction = program
            .expanded_bytecode
            .iter_mut()
            .find(|instruction| **instruction == executed)
            .expect("executed instruction in program");
        instruction.operands.imm ^= 1;
        assert!(matches!(
            backend.trace(&program, inputs.clone()),
            Err(TraceError::BytecodeInstructionMismatch { .. })
        ));
        let summary = backend
            .execute(&program, inputs, 100)
            .expect("execute failed");
        assert!(matches!(
            backend.replay_chunk(&summary.checkpoints[0]),
            Err(TraceError::BytecodeInstructionMismatch { .. })
        ));
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
    use jolt_riscv::{FieldInlineOp, FIELD_INLINE_OPCODE, RV64IMAC_JOLT_FIELD_INLINE};

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
        let bytecode = BytecodePreprocessing::preprocess(
            vec![instruction.try_jolt_instruction_row().unwrap()],
            0x8000_0000,
            RV64IMAC_JOLT_FIELD_INLINE,
        )
        .unwrap();
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
    use jolt_riscv::{RamRead, RamWrite, RV64IMAC_JOLT};

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
                let bytecode =
                    BytecodePreprocessing::preprocess(instructions, 0x8000_0000, RV64IMAC_JOLT)
                        .unwrap();
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
                let mut mismatched_bytecode = bytecode.clone();
                mismatched_bytecode.bytecode[1].operands.imm = 8;
                assert!(matches!(
                    collect_rows(&cycles, &mismatched_bytecode),
                    Err(TraceError::BytecodeInstructionMismatch { pc: 1, .. })
                ));
                let empty_bytecode =
                    BytecodePreprocessing::preprocess(Vec::new(), 0x8000_0000, RV64IMAC_JOLT)
                        .unwrap();
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
