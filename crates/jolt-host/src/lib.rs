//! Host-side guest program compilation, decoding, and tracing.
//!
//! Provides [`Program`] for building guest RISC-V programs via the `jolt` CLI,
//! decoding ELFs into instructions, and tracing execution to produce [`Cycle`]
//! vectors for the proving pipeline.
//!
//! This crate is independent of the proving system — it depends only on
//! `common` (memory config), `tracer` (RISC-V emulation), and standard I/O.

mod analyze;
mod cycle_row;
mod cycle_row_impl;
mod program;

pub use cycle_row::CycleRow;

use std::path::{Path, PathBuf};

pub use analyze::ProgramSummary;
pub use program::decode;

pub use common::jolt_device::{JoltDevice, MemoryConfig};
pub use tracer::emulator::memory::Memory;
pub use tracer::instruction::{Cycle, Instruction};
pub use tracer::LazyTraceIterator;

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

/// Host-side builder for guest RISC-V programs.
///
/// Provides methods to configure, compile, decode, and trace guest ELF binaries.
/// Call [`Program::new`] with the guest crate name, optionally configure via
/// `set_*` methods, then use [`Program::build`], [`Program::decode`], or
/// [`Program::trace`] to compile and execute the guest program.
#[derive(Clone, Debug)]
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
    elf: Option<PathBuf>,
    elf_compute_advice: Option<PathBuf>,
}

impl Program {
    /// Returns the path to the built guest ELF, if available.
    pub fn elf_path(&self) -> Option<&Path> {
        self.elf.as_deref()
    }

    /// Returns the path to the built compute-advice ELF, if available.
    pub fn elf_compute_advice_path(&self) -> Option<&Path> {
        self.elf_compute_advice.as_deref()
    }
}
