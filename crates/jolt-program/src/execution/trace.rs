use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_riscv::NormalizedInstruction;

use super::{ExecutionBackend, TraceError, TraceSource};

/// Program data prepared for execution and preprocessing.
///
/// This is the stage after `DecodedProgramImage`: decoded instruction rows have
/// been expanded into the final Jolt bytecode, while the original ELF bytes are
/// still kept for execution backends that run the program from its ELF image.
#[derive(Debug, Clone, Default)]
pub struct ExecutableProgram {
    elf_bytes: Vec<u8>,
    /// Bytecode after expanding decoded RV64 instructions into executable Jolt rows.
    pub expanded_bytecode: Vec<NormalizedInstruction>,
    /// Initial byte values for memory-backed ELF sections.
    pub memory_init: Vec<(u64, u8)>,
    /// End address of the loaded program image.
    pub program_end: u64,
    /// ELF entry point.
    pub entry_address: u64,
}

impl ExecutableProgram {
    pub fn from_elf_bytes(elf_bytes: Vec<u8>) -> Self {
        Self {
            elf_bytes,
            expanded_bytecode: Vec::new(),
            memory_init: Vec::new(),
            program_end: 0,
            entry_address: 0,
        }
    }

    pub fn from_parts(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<NormalizedInstruction>,
        memory_init: Vec<(u64, u8)>,
        program_end: u64,
        entry_address: u64,
    ) -> Self {
        Self {
            elf_bytes,
            expanded_bytecode,
            memory_init,
            program_end,
            entry_address,
        }
    }

    /// Creates an executable from a decoded program image and its expanded bytecode.
    ///
    /// `DecodedProgramImage` contains the rows and memory decoded directly from
    /// the ELF. The caller supplies `expanded_bytecode`, which is the result of
    /// expanding those decoded rows into the bytecode used by Jolt.
    #[cfg(feature = "image")]
    pub fn from_decoded_image(
        elf_bytes: Vec<u8>,
        expanded_bytecode: Vec<NormalizedInstruction>,
        image: crate::image::DecodedProgramImage,
    ) -> Self {
        Self::from_parts(
            elf_bytes,
            expanded_bytecode,
            image.memory_init,
            image.program_end,
            image.entry_address,
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegisterRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegisterWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegisterState {
    pub rs1: Option<RegisterRead>,
    pub rs2: Option<RegisterRead>,
    pub rd: Option<RegisterWrite>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RamRead {
    pub address: u64,
    pub value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RamWrite {
    pub address: u64,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RamAccess {
    Read(RamRead),
    Write(RamWrite),
    #[default]
    NoOp,
}

#[derive(Default, Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryImage {
    pub bytes: Vec<(u64, u8)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TraceRow {
    pub instruction: NormalizedInstruction,
    pub registers: RegisterState,
    pub ram_access: RamAccess,
}

#[derive(Debug, Clone)]
pub struct TraceOutput<T> {
    pub trace: T,
    pub device: JoltDevice,
    pub final_memory: Option<MemoryImage>,
}

impl<T> TraceOutput<T> {
    pub fn new(trace: T, device: JoltDevice, final_memory: Option<MemoryImage>) -> Self {
        Self {
            trace,
            device,
            final_memory,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct OwnedTrace {
    rows: Vec<TraceRow>,
    next: usize,
}

impl OwnedTrace {
    pub fn new(rows: Vec<TraceRow>) -> Self {
        Self { rows, next: 0 }
    }

    pub fn rows(&self) -> &[TraceRow] {
        &self.rows
    }

    pub fn into_rows(self) -> Vec<TraceRow> {
        self.rows
    }
}

impl From<Vec<TraceRow>> for OwnedTrace {
    fn from(rows: Vec<TraceRow>) -> Self {
        Self::new(rows)
    }
}

impl TraceSource for OwnedTrace {
    fn next_row(&mut self) -> Option<TraceRow> {
        let row = self.rows.get(self.next).copied();
        self.next += usize::from(row.is_some());
        row
    }
}
