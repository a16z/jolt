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
    ExecutionBackend, JoltProgram, MemoryImage, OwnedTrace, RamAccess, RamRead, RamWrite,
    RegisterRead, RegisterState, RegisterWrite, TraceError, TraceInputs, TraceOutput, TraceRow,
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
        )
        .with_advice_tape(Some(record.advice_tape)))
    }
}

/// Result of a recording pass.
struct RecordRunOutput {
    observations: Vec<Observation>,
    device: JoltDevice,
    final_memory: MemoryImage,
    advice_tape: Vec<u8>,
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
