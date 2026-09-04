use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, ExecutionSummary, JoltProgram, MemoryImage,
    OwnedTrace, RamAccess as ProgramRamAccess, RamRead as ProgramRamRead,
    RamWrite as ProgramRamWrite, RegisterRead, RegisterState, RegisterWrite, TraceError,
    TraceInputs, TraceOutput, TraceRow,
};
use jolt_program::preprocess::BytecodePreprocessing;
use jolt_riscv::{JoltInstructionRow, JoltTraceRow};
use rayon::prelude::*;

use common::jolt_device::JoltDevice;

use crate::emulator::cpu::AdviceTape;
use crate::emulator::decode_cache::DecodeCache;
use crate::instruction::{Cycle, RAMAccess};
use crate::parallel::{ChunkCheckpoint, ChunkWorker, PassOne, SnapshotPool};
use crate::trace_row::{cycle_to_trace_row, CycleConversionError};

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

    /// Executes the program and builds proof rows directly, without first
    /// allocating the wider execution rows.
    pub fn trace_compact(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
        bytecode: &BytecodePreprocessing,
    ) -> Result<TraceOutput<Arc<Vec<JoltTraceRow>>>, CompactTraceError> {
        let execution = self.trace_execution(program, inputs)?;
        let mut rows = collect_rows(execution.cycles, |cycle| {
            cycle_to_trace_row(&cycle, bytecode)
        })?;
        while rows.last() == Some(&JoltTraceRow::default()) {
            rows.pop();
        }
        Ok(TraceOutput::new(
            Arc::new(rows),
            execution.device,
            Some(execution.final_memory),
            Some(execution.advice_tape),
        ))
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
}

#[derive(Debug, thiserror::Error)]
pub enum CompactTraceError {
    #[error(transparent)]
    Trace(#[from] TraceError),
    #[error(transparent)]
    Row(#[from] CycleConversionError),
}

struct TraceExecution {
    cycles: Vec<Cycle>,
    final_memory: MemoryImage,
    device: JoltDevice,
    advice_tape: Vec<u8>,
}

impl ExecutionBackend for TracerBackend {
    type Trace = OwnedTrace;

    fn trace(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError> {
        let execution = self.trace_execution(program, inputs)?;
        let rows = collect_rows(execution.cycles, trace_row_from_cycle)?;
        Ok(TraceOutput::new(
            OwnedTrace::new(rows),
            execution.device,
            Some(execution.final_memory),
            Some(execution.advice_tape),
        ))
    }
}

const PARALLEL_ROW_CONVERSION_THRESHOLD: usize = 1 << 14;

fn collect_rows<R, E>(
    cycles: Vec<Cycle>,
    convert: impl Fn(Cycle) -> Result<R, E> + Sync,
) -> Result<Vec<R>, E>
where
    R: Default + Send,
    E: Send + Sync,
{
    let parallel = cycles.len() > PARALLEL_ROW_CONVERSION_THRESHOLD;
    let _span = tracing::info_span!(
        "trace_rows_from_cycles",
        rows = cycles.len(),
        workers = if parallel {
            rayon::current_num_threads()
        } else {
            1
        }
    )
    .entered();
    if !parallel {
        return cycles.into_iter().map(convert).collect();
    }

    // Rayon's fallible collector creates temporary shard vectors. Capturing the
    // error out of band keeps collection indexed and writes into one allocation.
    let error = OnceLock::new();
    let rows = cycles
        .into_par_iter()
        .map(|cycle| match convert(cycle) {
            Ok(row) => row,
            Err(worker_error) => {
                let _ = error.set(worker_error);
                R::default()
            }
        })
        .collect();
    match error.into_inner() {
        Some(worker_error) => Err(worker_error),
        None => Ok(rows),
    }
}

/// A resume point for the chunked-execution contract, built on the two-pass
/// parallel machinery (PR #1717): a tick-boundary checkpoint and memory
/// image — captured together at the same boundary and shared, via `Arc`,
/// by every chunk that resumes there — plus this chunk's row window
/// relative to that boundary.
pub struct TracerChunkCheckpoint {
    /// Boundary CPU/MMU/device state.
    boundary: Arc<ChunkCheckpoint>,
    /// Full-size flat-memory image at the same boundary (SnapshotPool
    /// layout).
    image: Arc<Vec<u64>>,
    seed: Arc<WorkerSeed>,
    /// Rows to discard after resuming at the boundary.
    skip_rows: usize,
    /// Rows this chunk emits.
    take_rows: usize,
}

/// Static per-program worker seed, shared by every checkpoint.
struct WorkerSeed {
    device: Option<JoltDevice>,
    decode: DecodeCache,
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
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        if chunk_size == 0 {
            return Err(TraceError::Backend("chunk_size must be nonzero"));
        }

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
        });

        // Construction-only bookkeeping: a captured boundary plus the row
        // count pass-1 had produced there (used below to pick each mark's
        // resume boundary; not needed at replay time).
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

        // The fast pass: execute mode, no rows; capture a boundary checkpoint
        // whenever a chunk mark is crossed (subject to the spacing floor).
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

        // One contract checkpoint per exact chunk mark, resuming from the
        // latest boundary at or before the mark.
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

    fn replay_chunk(&self, checkpoint: &Self::Checkpoint) -> Result<Self::Trace, TraceError> {
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

        let rows = cycles[checkpoint.skip_rows..needed]
            .iter()
            .map(|cycle| trace_row_from_cycle(*cycle))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(OwnedTrace::new(rows))
    }
}

fn trace_row_from_cycle(cycle: Cycle) -> Result<TraceRow, TraceError> {
    let row = TraceRow::new(
        jolt_instruction_row(&cycle)?,
        register_state(&cycle),
        cycle.ram_access().into(),
    )?;
    #[cfg(feature = "field-inline")]
    let row = {
        let mut row = row;
        row.field_inline = cycle.field_inline_trace().map(Into::into);
        row
    };
    Ok(row)
}

fn jolt_instruction_row(cycle: &Cycle) -> Result<JoltInstructionRow, TraceError> {
    let instruction = cycle.instruction();
    instruction
        .try_jolt_instruction_row()
        .map_err(|_| TraceError::Backend("execution trace contained a source-only instruction"))
}

fn register_state(cycle: &Cycle) -> RegisterState {
    RegisterState {
        rs1: cycle
            .rs1_read()
            .map(|(register, value)| RegisterRead { register, value }),
        rs2: cycle
            .rs2_read()
            .map(|(register, value)| RegisterRead { register, value }),
        rd: cycle
            .rd_write()
            .map(|(register, pre_value, post_value)| RegisterWrite {
                register,
                pre_value,
                post_value,
            }),
    }
}

impl From<RAMAccess> for ProgramRamAccess {
    fn from(access: RAMAccess) -> Self {
        match access {
            RAMAccess::Read(read) => Self::Read(ProgramRamRead {
                address: read.address,
                value: read.value,
            }),
            RAMAccess::Write(write) => Self::Write(ProgramRamWrite {
                address: write.address,
                pre_value: write.pre_value,
                post_value: write.post_value,
            }),
            RAMAccess::NoOp => Self::NoOp,
        }
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
            let mut replayed: Vec<Vec<TraceRow>> = summary
                .checkpoints
                .iter()
                .rev()
                .map(|checkpoint| {
                    backend
                        .replay_chunk(checkpoint)
                        .expect("replay failed")
                        .into_rows()
                })
                .collect();
            replayed.reverse();

            let last = replayed.len() - 1;
            for (i, rows) in replayed.iter().enumerate() {
                if i < last {
                    assert_eq!(rows.len(), chunk_size, "chunk_size {chunk_size}, chunk {i}");
                }
            }
            let concat: Vec<TraceRow> = replayed.into_iter().flatten().collect();
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
            .into_rows();
        assert_eq!(first.as_slice(), &eager_rows[..CHUNK_SIZE]);

        let last_mark = (summary.checkpoints.len() - 1) * CHUNK_SIZE;
        let last = backend
            .replay_chunk(summary.checkpoints.last().expect("nonempty trace"))
            .expect("replay failed")
            .into_rows();
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
    #[cfg(feature = "field-inline")]
    use crate::{
        emulator::{cpu::Cpu, default_terminal::DefaultTerminal},
        instruction::Instruction,
    };
    #[cfg(feature = "field-inline")]
    use jolt_program::field_inline::{FieldEncodedValue, FieldInlineBridge};
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
    fn trace_row_from_cycle_carries_field_inline_payload() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.write_register(5, 11);
        let instruction = Instruction::decode(
            field_inline_word(FieldInlineOp::LoadFromX, 2, 5, 0),
            0x8000_0000,
            false,
        )
        .unwrap();
        let mut trace = Vec::new();
        instruction.trace(&mut cpu, Some(&mut trace));
        assert_eq!(trace.len(), 1);

        let row = super::trace_row_from_cycle(trace.remove(0)).unwrap();
        assert_eq!(row.rs1_read().unwrap().register, 5);
        assert_eq!(row.rs1_read().unwrap().value, 11);
        assert!(row.rs2_read().is_none());
        assert!(row.rd_write().is_none());
        let field_trace = row.field_inline.unwrap();
        assert_eq!(field_trace.op, Some(FieldInlineOp::LoadFromX));
        assert_eq!(
            field_trace.bridge,
            Some(FieldInlineBridge::LoadFromX {
                x_register: 5,
                x_value: 11,
                field_value: FieldEncodedValue::from_u64(11),
            })
        );
    }
}
