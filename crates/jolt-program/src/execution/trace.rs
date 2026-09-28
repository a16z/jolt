use std::sync::Arc;

use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow, RV64IMAC_JOLT};

use super::{ExecutionBackend, TraceError, TraceSource};

mod row;

pub use row::{
    RamAccess, RamRead, RamWrite, RegisterRead, RegisterState, RegisterWrite, TraceRow,
    TraceRowError,
};

/// A Jolt-ready program built from an RV64 ELF image.
///
/// This is the stage after `Rv64ProgramImage`: decoded RV64 instruction rows
/// have been expanded into the bytecode used by Jolt preprocessing, while the
/// original ELF bytes are still kept for backends that run the source program
/// from its ELF image.
#[derive(Debug, Clone)]
pub struct JoltProgram {
    elf_bytes: Vec<u8>,
    /// Final Jolt bytecode rows after expanding decoded RV64 instructions.
    pub expanded_bytecode: Vec<JoltInstructionRow>,
    /// Initial byte values for memory-backed ELF sections.
    pub memory_init: Vec<(u64, u8)>,
    /// End address of the loaded program image.
    pub program_end: u64,
    /// ELF entry point.
    pub entry_address: u64,
    /// Selected instruction legality/profile for this program.
    pub profile: JoltInstructionProfile,
}

impl Default for JoltProgram {
    fn default() -> Self {
        Self::from_elf_bytes(Vec::new())
    }
}

impl JoltProgram {
    pub fn from_elf_bytes(elf_bytes: Vec<u8>) -> Self {
        Self {
            elf_bytes,
            expanded_bytecode: Vec::new(),
            memory_init: Vec::new(),
            program_end: 0,
            entry_address: 0,
            profile: RV64IMAC_JOLT,
        }
    }

    pub fn from_parts(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        program_end: u64,
        entry_address: u64,
    ) -> Self {
        Self::from_parts_with_profile(
            elf_bytes,
            expanded_bytecode,
            memory_init,
            program_end,
            entry_address,
            RV64IMAC_JOLT,
        )
    }

    pub fn from_parts_with_profile(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        program_end: u64,
        entry_address: u64,
        profile: JoltInstructionProfile,
    ) -> Self {
        Self {
            elf_bytes,
            expanded_bytecode,
            memory_init,
            program_end,
            entry_address,
            profile,
        }
    }

    /// Creates a Jolt program from an RV64 program image and its expanded bytecode.
    ///
    /// `Rv64ProgramImage` contains the rows and memory decoded directly from
    /// the ELF. The caller supplies `expanded_bytecode`, which is the result of
    /// expanding those decoded rows into the bytecode used by Jolt.
    #[cfg(feature = "image")]
    pub fn from_rv64_image(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        image: crate::image::Rv64ProgramImage,
    ) -> Self {
        Self::from_rv64_image_with_profile(elf_bytes, expanded_bytecode, image, RV64IMAC_JOLT)
    }

    #[cfg(feature = "image")]
    pub fn from_rv64_image_with_profile(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<JoltInstructionRow>,
        image: crate::image::Rv64ProgramImage,
        profile: JoltInstructionProfile,
    ) -> Self {
        Self::from_parts_with_profile(
            elf_bytes,
            expanded_bytecode,
            image.memory_init,
            image.program_end,
            image.entry_address,
            profile,
        )
    }

    pub fn elf_bytes(&self) -> &[u8] {
        &self.elf_bytes
    }

    pub fn trace_with<B: ExecutionBackend>(
        &self,
        backend: &mut B,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<B::Trace>, TraceError> {
        backend.trace(self, inputs)
    }
}

#[derive(Default, Debug, Clone)]
pub struct TraceInputs {
    pub inputs: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub trusted_advice: Vec<u8>,
    pub memory_config: MemoryConfig,
    /// Runtime advice tape to seed execution with (the SDK's two-pass advice
    /// flow: pass 1 populates the tape, pass 2 consumes it). Read cursor
    /// always starts at 0.
    pub advice_tape: Option<Vec<u8>>,
}

impl TraceInputs {
    pub fn new(
        inputs: Vec<u8>,
        untrusted_advice: Vec<u8>,
        trusted_advice: Vec<u8>,
        memory_config: MemoryConfig,
    ) -> Self {
        Self {
            inputs,
            untrusted_advice,
            trusted_advice,
            memory_config,
            advice_tape: None,
        }
    }

    pub fn with_advice_tape(mut self, advice_tape: Option<Vec<u8>>) -> Self {
        self.advice_tape = advice_tape;
        self
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MemoryImage {
    pub bytes: Vec<(u64, u8)>,
}

#[derive(Debug, Clone)]
pub struct TraceOutput<T> {
    pub trace: T,
    pub device: JoltDevice,
    pub final_memory: Option<MemoryImage>,
    /// The populated runtime advice tape captured at guest termination
    /// (`None` when the backend produced no tape).
    pub advice_tape: Option<Vec<u8>>,
}

impl<T> TraceOutput<T> {
    /// `advice_tape` is a required parameter so that a backend (or a
    /// rebuild of an existing output) cannot silently discard a populated
    /// tape — the seam this field exists to plug.
    pub fn new(
        trace: T,
        device: JoltDevice,
        final_memory: Option<MemoryImage>,
        advice_tape: Option<Vec<u8>>,
    ) -> Self {
        Self {
            trace,
            device,
            final_memory,
            advice_tape,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct OwnedTrace {
    rows: Arc<Vec<TraceRow>>,
    next: usize,
}

impl OwnedTrace {
    pub fn new(rows: Vec<TraceRow>) -> Self {
        Self {
            rows: Arc::new(rows),
            next: 0,
        }
    }

    pub fn rows(&self) -> &[TraceRow] {
        self.rows.as_slice()
    }

    pub fn into_rows(self) -> Vec<TraceRow> {
        match Arc::try_unwrap(self.rows) {
            Ok(rows) => rows,
            Err(rows) => rows.as_ref().clone(),
        }
    }
}

impl From<Vec<TraceRow>> for OwnedTrace {
    fn from(rows: Vec<TraceRow>) -> Self {
        Self::new(rows)
    }
}

impl TraceSource for OwnedTrace {
    fn next_row(&mut self) -> Option<TraceRow> {
        #[cfg(not(feature = "field-inline"))]
        let row = self.rows.get(self.next).copied();
        #[cfg(feature = "field-inline")]
        let row = self.rows.get(self.next).cloned();
        self.next += usize::from(row.is_some());
        row
    }

    fn rows(&self) -> Option<&[TraceRow]> {
        (self.next == 0).then(|| self.rows.as_slice())
    }
}
