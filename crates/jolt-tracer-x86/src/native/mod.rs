//! Native (x86_64-linux) implementation: AOT compilation of expanded
//! bytecode and the execution driver.

mod compile;
#[doc(hidden)]
pub mod harness;
mod helpers;
mod memory;
mod state;

use std::sync::Arc;

use common::jolt_device::JoltDevice;
use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, ExecutionSummary, JoltProgram, MemoryImage,
    OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead, RegisterState, RegisterWrite,
    TraceError, TraceInputs, TraceOutput, TraceRow,
};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow};

use compile::CompiledProgram;
use memory::MemoryPlane;
use state::{ExitReason, GuestState, HostContext, Observation};

/// AOT x86-64 transpiling execution backend.
///
/// Compiles a [`JoltProgram`]'s expanded bytecode to native code on first use
/// and caches the artifact keyed by the program's identity, so repeated
/// traces and chunk replays reuse it.
#[derive(Default)]
pub struct X86TracerBackend {
    cache: Option<CachedProgram>,
}

struct CachedProgram {
    fingerprint: u64,
    compiled: Arc<CompiledProgram>,
}

/// Result of a fast (non-recording) pass.
pub struct FastRunOutput {
    pub trace_len: usize,
    pub device: JoltDevice,
    pub final_memory: MemoryImage,
    pub advice_tape: Vec<u8>,
}

impl X86TracerBackend {
    pub fn new() -> Self {
        Self::default()
    }

    fn compiled(&mut self, program: &JoltProgram) -> Result<Arc<CompiledProgram>, TraceError> {
        let fingerprint = fingerprint(program.elf_bytes());
        if let Some(cached) = &self.cache {
            if cached.fingerprint == fingerprint {
                return Ok(Arc::clone(&cached.compiled));
            }
        }
        let compiled = Arc::new(compile::compile(program)?);
        self.cache = Some(CachedProgram {
            fingerprint,
            compiled: Arc::clone(&compiled),
        });
        Ok(compiled)
    }

    /// Recording pass: execute with per-row value capture into an
    /// observation buffer sized to `expected_rows` (plus slack, so a
    /// divergence overflows loudly instead of writing out of bounds).
    fn record_run(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
        expected_rows: usize,
    ) -> Result<RecordRunOutput, TraceError> {
        let compiled = self.compiled(program)?;

        let mut device = JoltDevice::new(&inputs.memory_config);
        device.inputs.clone_from(&inputs.inputs);
        device.trusted_advice.clone_from(&inputs.trusted_advice);
        device.untrusted_advice.clone_from(&inputs.untrusted_advice);

        let plane_size = device.memory_layout.get_total_memory_size();
        let mut plane = MemoryPlane::new(plane_size as usize)?;
        plane.init_from_image(&program.memory_init)?;

        let mut host = HostContext {
            device,
            advice_tape: inputs.advice_tape.clone().unwrap_or_default(),
            advice_cursor: 0,
            helper_error: None,
        };

        let mut observations = vec![Observation::default(); expected_rows + 1];
        let obs_start = observations.as_mut_ptr();
        // SAFETY: one-past-the-end of the allocation, only ever compared
        // against, never dereferenced.
        let obs_end = unsafe { obs_start.add(observations.len()) };

        let mut guest = Box::new(GuestState {
            x: [0; common::constants::REGISTER_COUNT as usize],
            pc: program.entry_address,
            trace_len: 0,
            exit: ExitReason::Running as u64,
            fault_addr: 0,
            mem_base: plane.base() as u64,
            mem_size: plane.size() as u64,
            host: &raw mut host,
            advice_slots: [0; crate::native::state::ADVICE_SLOTS],
            advice_jobs: compiled.advice_jobs_ptr(),
            row_limit: u64::MAX,
            obs_cursor: obs_start,
            obs_end,
        });

        compiled.run_record(&mut guest)?;
        check_exit(&guest, &mut host)?;

        // The cursor's advance is the recorded row count.
        let recorded =
            (guest.obs_cursor as usize - obs_start as usize) / core::mem::size_of::<Observation>();
        observations.truncate(recorded);

        Ok(RecordRunOutput {
            observations,
            device: host.device,
            final_memory: MemoryImage {
                bytes: plane.materialized_nonzero_bytes(),
            },
            advice_tape: host.advice_tape,
        })
    }

    /// Fast pass: run the program to completion without materializing trace
    /// rows. (Checkpoint logging joins in the chunked-execution slice.)
    pub fn fast_run(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<FastRunOutput, TraceError> {
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        let compiled = self.compiled(program)?;

        let mut device = JoltDevice::new(&inputs.memory_config);
        device.inputs.clone_from(&inputs.inputs);
        device.trusted_advice.clone_from(&inputs.trusted_advice);
        device.untrusted_advice.clone_from(&inputs.untrusted_advice);

        let plane_size = device.memory_layout.get_total_memory_size();
        let mut plane = MemoryPlane::new(plane_size as usize)?;
        plane.init_from_image(&program.memory_init)?;

        let mut host = HostContext {
            device,
            advice_tape: inputs.advice_tape.clone().unwrap_or_default(),
            advice_cursor: 0,
            helper_error: None,
        };

        let mut guest = Box::new(GuestState {
            x: [0; common::constants::REGISTER_COUNT as usize],
            pc: program.entry_address,
            trace_len: 0,
            exit: ExitReason::Running as u64,
            fault_addr: 0,
            mem_base: plane.base() as u64,
            mem_size: plane.size() as u64,
            host: &raw mut host,
            advice_slots: [0; crate::native::state::ADVICE_SLOTS],
            advice_jobs: compiled.advice_jobs_ptr(),
            row_limit: u64::MAX,
            obs_cursor: core::ptr::null_mut(),
            obs_end: core::ptr::null_mut(),
        });

        compiled.run(&mut guest)?;
        check_exit(&guest, &mut host)?;

        Ok(FastRunOutput {
            trace_len: guest.trace_len as usize,
            device: host.device,
            final_memory: MemoryImage {
                bytes: plane.materialized_nonzero_bytes(),
            },
            advice_tape: host.advice_tape,
        })
    }
}

/// Translate a generated-code exit into a `TraceError`.
fn check_exit(guest: &GuestState, host: &mut HostContext) -> Result<(), TraceError> {
    match guest.exit {
        e if e == ExitReason::Terminated as u64 => {}
        e if e == ExitReason::FaultOutOfBounds as u64 => {
            return Err(TraceError::Backend("guest RAM access out of bounds"));
        }
        e if e == ExitReason::FaultBadJumpTarget as u64 => {
            return Err(TraceError::Backend(
                "indirect jump to a non-compiled address",
            ));
        }
        e if e == ExitReason::FaultObservationOverflow as u64 => {
            return Err(TraceError::Backend(
                "record pass overflowed the observation buffer (row-count divergence)",
            ));
        }
        _ => {
            if let Some(message) = host.helper_error.take() {
                tracing_error(&message);
            }
            return Err(TraceError::Backend("host helper reported an error"));
        }
    }
    Ok(())
}

#[expect(clippy::print_stderr)]
fn tracing_error(message: &str) {
    eprintln!("jolt-tracer-x86 helper error: {message}");
}

impl ExecutionBackend for X86TracerBackend {
    type Trace = OwnedTrace;

    /// Record mode: a fast pass sizes the observation buffer exactly, then the
    /// record body fills it and a Rust pass reassembles `TraceRow`s.
    ///
    /// Two passes cost about 12% over recording alone (the fast pass runs at
    /// several hundred MHz) and buy an exactly-sized buffer plus a
    /// cross-check: if the record body emits a different row count than the
    /// fast pass counted, the two diverged and that is a bug, caught here
    /// rather than in a proof.
    fn trace(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError> {
        let expected = self.fast_run(program, inputs.clone())?;
        let record = self.record_run(program, inputs, expected.trace_len)?;

        if record.observations.len() != expected.trace_len {
            return Err(TraceError::Backend(
                "record pass emitted a different row count than the fast pass",
            ));
        }
        let rows = reassemble_rows(&program.expanded_bytecode, &record.observations)?;
        Ok(TraceOutput::new(
            OwnedTrace::new(rows),
            record.device,
            Some(record.final_memory),
            Some(record.advice_tape),
        ))
    }
}

/// Result of a recording pass.
struct RecordRunOutput {
    observations: Vec<Observation>,
    device: JoltDevice,
    final_memory: MemoryImage,
    advice_tape: Vec<u8>,
}

/// Longest run of rows sharing a source address, i.e. the largest expansion
/// a single source instruction produces.
fn longest_group(bytecode: &[JoltInstructionRow]) -> usize {
    let mut longest = 1usize;
    let mut current = 1usize;
    for pair in bytecode.windows(2) {
        if pair[0].address == pair[1].address {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 1;
        }
    }
    longest
}

/// Rebuild `TraceRow`s from the static bytecode plus the recorded dynamic
/// values. Generated code cannot construct `TraceRow` directly (its `Option`
/// fields have no guaranteed layout), so this is the seam between the two.
fn reassemble_rows(
    bytecode: &[JoltInstructionRow],
    observations: &[Observation],
) -> Result<Vec<TraceRow>, TraceError> {
    let mut rows = Vec::with_capacity(observations.len());
    for observation in observations {
        let row = bytecode
            .get(observation.row_index as usize)
            .ok_or(TraceError::Backend("observation row index out of range"))?;
        rows.push(TraceRow {
            instruction: *row,
            registers: RegisterState {
                rs1: register_read(row.operands.rs1, observation.rs1),
                rs2: register_read(row.operands.rs2, observation.rs2),
                rd: row.operands.rd.map(|register| RegisterWrite {
                    register,
                    // x0 reads as zero on both sides of a write.
                    pre_value: if register == 0 { 0 } else { observation.rd_pre },
                    post_value: if register == 0 {
                        0
                    } else {
                        observation.rd_post
                    },
                }),
            },
            ram_access: ram_access(row.instruction_kind, observation),
            #[cfg(feature = "field-inline")]
            field_inline: None,
        });
    }
    Ok(rows)
}

fn register_read(register: Option<u8>, value: u64) -> Option<RegisterRead> {
    register.map(|register| RegisterRead {
        register,
        value: if register == 0 { 0 } else { value },
    })
}

/// Which RAM access a row records is a static property of its kind: only
/// `Ld` and `Sd` touch RAM in final bytecode.
fn ram_access(kind: JoltInstructionKind, observation: &Observation) -> RamAccess {
    match kind {
        JoltInstructionKind::LD => RamAccess::Read(RamRead {
            address: observation.ram_address,
            value: observation.ram_pre,
        }),
        JoltInstructionKind::SD => RamAccess::Write(RamWrite {
            address: observation.ram_address,
            pre_value: observation.ram_pre,
            post_value: observation.ram_post,
        }),
        _ => RamAccess::NoOp,
    }
}

/// FNV-1a over the ELF bytes: cheap, stable identity for the compile cache.
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A resume point for the chunked contract: the guest state at a group
/// boundary plus the memory image needed to replay from it.
///
/// The spec's preferred design is an access-value log (replay answers reads
/// from the log, needing no image). This implementation snapshots the memory
/// plane instead, which is the same tradeoff #1717's parallel machinery
/// makes: correctness first, and the image is what makes a chunk replayable
/// with no dependence on any other chunk. The `Checkpoint` associated type
/// is deliberately opaque, so switching to logs later changes nothing for
/// consumers.
pub struct X86Checkpoint {
    compiled: Arc<CompiledProgram>,
    /// The program's expanded bytecode, shared with every checkpoint: row
    /// reassembly needs the static half of each row.
    bytecode: Arc<Vec<JoltInstructionRow>>,
    /// The resume state, shared by every chunk that resumes from it. Boundary
    /// images are plane-sized (tens of MB), so sharing is not an optimization
    /// but a requirement: one image per chunk would exhaust memory at small
    /// chunk sizes.
    boundary: Arc<Boundary>,
    /// Rows to discard after resuming (the boundary may precede the mark).
    skip_rows: usize,
    /// Rows this chunk emits.
    take_rows: usize,
    /// Longest source-instruction group in the program. Replay can only stop
    /// at a group boundary, so it may overshoot its window by up to this
    /// much; the observation buffer carries that slack.
    max_group_rows: usize,
}

/// Guest state at a group boundary: everything a replay needs to resume.
struct Boundary {
    registers: [u64; common::constants::REGISTER_COUNT as usize],
    pc: u64,
    /// Full plane image at the boundary.
    memory: Vec<u8>,
    memory_config: common::jolt_device::MemoryConfig,
    device_inputs: Vec<u8>,
    device_trusted_advice: Vec<u8>,
    device_untrusted_advice: Vec<u8>,
    device_outputs: Vec<u8>,
    device_panic: bool,
    advice_tape: Vec<u8>,
    advice_cursor: usize,
}

// SAFETY: every field is owned data or an Arc to an immutable artifact (the
// compiled code and the bytecode), so the checkpoint can move between
// threads.
unsafe impl Send for X86Checkpoint {}
// SAFETY: replay only reads the checkpoint, copying its image and registers
// into fresh per-replay state, so shared references are safe to use
// concurrently; there is no interior mutability.
unsafe impl Sync for X86Checkpoint {}

impl ChunkedExecutionBackend for X86TracerBackend {
    type Checkpoint = X86Checkpoint;

    /// Fast pass with checkpoint capture: run in row-bounded increments,
    /// pausing at the first group boundary at or past each mark and
    /// snapshotting the state there.
    fn execute(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
        chunk_size: usize,
    ) -> Result<ExecutionSummary<Self::Checkpoint>, TraceError> {
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        if chunk_size == 0 {
            return Err(TraceError::Backend("chunk_size must be nonzero"));
        }
        let compiled = self.compiled(program)?;

        let mut device = JoltDevice::new(&inputs.memory_config);
        device.inputs.clone_from(&inputs.inputs);
        device.trusted_advice.clone_from(&inputs.trusted_advice);
        device.untrusted_advice.clone_from(&inputs.untrusted_advice);

        let plane_size = device.memory_layout.get_total_memory_size();
        let mut plane = MemoryPlane::new(plane_size as usize)?;
        plane.init_from_image(&program.memory_init)?;

        let mut host = HostContext {
            device,
            advice_tape: inputs.advice_tape.clone().unwrap_or_default(),
            advice_cursor: 0,
            helper_error: None,
        };
        let mut guest = Box::new(GuestState {
            x: [0; common::constants::REGISTER_COUNT as usize],
            pc: program.entry_address,
            trace_len: 0,
            exit: ExitReason::Running as u64,
            fault_addr: 0,
            mem_base: plane.base() as u64,
            mem_size: plane.size() as u64,
            host: &raw mut host,
            advice_slots: [0; crate::native::state::ADVICE_SLOTS],
            advice_jobs: compiled.advice_jobs_ptr(),
            row_limit: 0,
            obs_cursor: core::ptr::null_mut(),
            obs_end: core::ptr::null_mut(),
        });

        // Boundaries: (rows_before, snapshot). The first is the program's
        // initial state, which every early chunk resumes from.
        let mut boundaries: Vec<(usize, Arc<Boundary>)> = Vec::new();
        let bytecode = Arc::new(program.expanded_bytecode.clone());
        let max_group_rows = longest_group(&bytecode);
        let snapshot = |guest: &GuestState, host: &HostContext, plane: &MemoryPlane| {
            Arc::new(Boundary {
                registers: guest.x,
                pc: guest.pc,
                memory: plane.to_vec(),
                memory_config: inputs.memory_config,
                device_inputs: host.device.inputs.clone(),
                device_trusted_advice: host.device.trusted_advice.clone(),
                device_untrusted_advice: host.device.untrusted_advice.clone(),
                device_outputs: host.device.outputs.clone(),
                device_panic: host.device.panic,
                advice_tape: host.advice_tape.clone(),
                advice_cursor: host.advice_cursor,
            })
        };
        boundaries.push((0, snapshot(&guest, &host, &plane)));

        // Run in increments, capturing a boundary at each pause. Checkpoints
        // carry a full image, so they are spaced at least this far apart
        // regardless of how small chunk_size is; `skip_rows` covers the gap.
        const MIN_SPACING_ROWS: usize = 1 << 16;
        let spacing = chunk_size.max(MIN_SPACING_ROWS);
        loop {
            guest.row_limit = (guest.trace_len as usize + spacing) as u64;
            compiled.run_pausable(&mut guest)?;
            if guest.exit == ExitReason::Paused as u64 {
                guest.exit = ExitReason::Running as u64;
                boundaries.push((guest.trace_len as usize, snapshot(&guest, &host, &plane)));
                continue;
            }
            check_exit(&guest, &mut host)?;
            break;
        }
        let trace_len = guest.trace_len as usize;

        // One contract checkpoint per chunk mark, resuming from the latest
        // boundary at or before it.
        let mut checkpoints = Vec::with_capacity(trace_len.div_ceil(chunk_size));
        let mut index = 0usize;
        for chunk in 0..trace_len.div_ceil(chunk_size) {
            let mark = chunk * chunk_size;
            while index + 1 < boundaries.len() && boundaries[index + 1].0 <= mark {
                index += 1;
            }
            let (rows_before, boundary) = &boundaries[index];
            checkpoints.push(X86Checkpoint {
                compiled: Arc::clone(&compiled),
                bytecode: Arc::clone(&bytecode),
                boundary: Arc::clone(boundary),
                skip_rows: mark - rows_before,
                take_rows: chunk_size.min(trace_len - mark),
                max_group_rows,
            });
        }

        Ok(ExecutionSummary {
            checkpoints,
            trace_len,
            device: host.device,
            final_memory: Some(MemoryImage {
                bytes: plane.materialized_nonzero_bytes(),
            }),
            advice_tape: Some(host.advice_tape),
        })
    }

    /// Replay one chunk in record mode from its checkpoint, discarding the
    /// leading rows the boundary precedes and keeping exactly this chunk's.
    fn replay_chunk(&self, checkpoint: &Self::Checkpoint) -> Result<Self::Trace, TraceError> {
        let boundary = &checkpoint.boundary;
        let mut device = JoltDevice::new(&boundary.memory_config);
        device.inputs.clone_from(&boundary.device_inputs);
        device
            .trusted_advice
            .clone_from(&boundary.device_trusted_advice);
        device
            .untrusted_advice
            .clone_from(&boundary.device_untrusted_advice);
        device.outputs.clone_from(&boundary.device_outputs);
        device.panic = boundary.device_panic;

        let mut plane = MemoryPlane::new(boundary.memory.len())?;
        plane.restore(&boundary.memory);

        let mut host = HostContext {
            device,
            advice_tape: boundary.advice_tape.clone(),
            advice_cursor: boundary.advice_cursor,
            helper_error: None,
        };

        let needed = checkpoint.skip_rows + checkpoint.take_rows;
        // Slack for the overshoot: the row budget is checked at group starts,
        // so a replay stops at the first boundary at or past `needed`.
        let mut observations = vec![Observation::default(); needed + checkpoint.max_group_rows + 1];
        let obs_start = observations.as_mut_ptr();
        // SAFETY: one past the end of the allocation; compared, never read.
        let obs_end = unsafe { obs_start.add(observations.len()) };

        let mut guest = Box::new(GuestState {
            x: boundary.registers,
            pc: boundary.pc,
            trace_len: 0,
            exit: ExitReason::Running as u64,
            fault_addr: 0,
            mem_base: plane.base() as u64,
            mem_size: plane.size() as u64,
            host: &raw mut host,
            advice_slots: [0; crate::native::state::ADVICE_SLOTS],
            advice_jobs: checkpoint.compiled.advice_jobs_ptr(),
            // Stop at the first group boundary at or past the window.
            row_limit: needed as u64,
            obs_cursor: obs_start,
            obs_end,
        });

        checkpoint.compiled.run_record_pausable(&mut guest)?;
        if guest.exit != ExitReason::Paused as u64 {
            check_exit(&guest, &mut host)?;
        }

        let recorded =
            (guest.obs_cursor as usize - obs_start as usize) / core::mem::size_of::<Observation>();
        if recorded < needed {
            return Err(TraceError::Backend(
                "chunk replay produced fewer rows than the chunk window",
            ));
        }
        observations.truncate(needed);
        let rows = reassemble_rows(&checkpoint.bytecode, &observations[checkpoint.skip_rows..])?;
        Ok(OwnedTrace::new(rows))
    }
}
