//! Host-side guest program compilation, decoding, and tracing.
//!
//! Provides [`Program`] for building guest RISC-V programs via the `jolt` CLI,
//! decoding ELFs into instructions, and tracing execution to produce [`Cycle`]
//! vectors for the proving pipeline.
//!
//! This crate is independent of the proving system — it depends only on
//! `common` (memory config), `tracer` (RISC-V emulation), and standard I/O.

mod analyze;
mod program;

pub use analyze::ProgramSummary;
pub use program::{decode, trace, trace_to_file};

// Re-export types that callers need
pub use common::jolt_device::{JoltDevice, MemoryConfig};
pub use tracer::emulator::memory::Memory;
pub use tracer::instruction::{Cycle, Instruction};
pub use tracer::LazyTraceIterator;

use std::path::PathBuf;

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    profile: Option<String>,
    heap_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_untrusted_advice_size: u64,
    max_trusted_advice_size: u64,
    max_output_size: u64,
    std: bool,
    backtrace: Option<String>,
    pub elf: Option<PathBuf>,
    pub elf_compute_advice: Option<PathBuf>,
}
