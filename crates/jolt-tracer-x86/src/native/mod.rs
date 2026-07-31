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
    ExecutionBackend, JoltProgram, MemoryImage, OwnedTrace, TraceError, TraceInputs, TraceOutput,
};

use compile::CompiledProgram;
use memory::MemoryPlane;
use state::{ExitReason, GuestState, HostContext};

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
        });

        compiled.run(&mut guest)?;

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
            _ => {
                if let Some(message) = host.helper_error.take() {
                    tracing_error(&message);
                }
                return Err(TraceError::Backend("host helper reported an error"));
            }
        }

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

#[expect(clippy::print_stderr)]
fn tracing_error(message: &str) {
    eprintln!("jolt-tracer-x86 helper error: {message}");
}

impl ExecutionBackend for X86TracerBackend {
    type Trace = OwnedTrace;

    fn trace(
        &mut self,
        _program: &JoltProgram,
        _inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError> {
        // Record mode lands in the full-coverage slice; fail fast until then
        // so no caller silently gets an empty trace.
        Err(TraceError::Backend(
            "jolt-tracer-x86 record mode is not implemented yet (use fast_run)",
        ))
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
