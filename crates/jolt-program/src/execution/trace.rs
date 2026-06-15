use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_riscv::{JoltCycle, JoltInstructionProfile, JoltInstructionRow, RV64IMAC_JOLT};
use std::sync::Arc;

#[cfg(feature = "field-inline")]
use crate::field_inline::FieldInlineTraceData;

use super::{ExecutionBackend, TraceError, TraceSource};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RegisterRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RegisterWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RegisterState {
    pub rs1: Option<RegisterRead>,
    pub rs2: Option<RegisterRead>,
    pub rd: Option<RegisterWrite>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RamRead {
    pub address: u64,
    pub value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RamWrite {
    pub address: u64,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum RamAccess {
    Read(RamRead),
    Write(RamWrite),
    #[default]
    NoOp,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MemoryImage {
    pub bytes: Vec<(u64, u8)>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "field-inline"), derive(Copy))]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct TraceRow {
    pub instruction: JoltInstructionRow,
    pub registers: RegisterState,
    pub ram_access: RamAccess,
    #[cfg(feature = "field-inline")]
    pub field_inline: Option<Arc<FieldInlineTraceData>>,
}

impl JoltCycle for TraceRow {
    type Instruction = JoltInstructionRow;

    #[inline]
    fn instruction(&self) -> Self::Instruction {
        self.instruction
    }

    #[inline]
    fn rs1_val(&self) -> Option<u64> {
        self.registers.rs1.map(|read| read.value)
    }

    #[inline]
    fn rs2_val(&self) -> Option<u64> {
        self.registers.rs2.map(|read| read.value)
    }

    #[inline]
    fn rd_vals(&self) -> Option<(u64, u64)> {
        self.registers
            .rd
            .map(|write| (write.pre_value, write.post_value))
    }

    #[inline]
    fn ram_access_address(&self) -> Option<u64> {
        match self.ram_access {
            RamAccess::Read(read) => Some(read.address),
            RamAccess::Write(write) => Some(write.address),
            RamAccess::NoOp => None,
        }
    }

    #[inline]
    fn ram_read_value(&self) -> Option<u64> {
        match self.ram_access {
            RamAccess::Read(read) => Some(read.value),
            RamAccess::Write(write) => Some(write.pre_value),
            RamAccess::NoOp => None,
        }
    }

    #[inline]
    fn ram_write_value(&self) -> Option<u64> {
        match self.ram_access {
            RamAccess::Write(write) => Some(write.post_value),
            RamAccess::Read(_) | RamAccess::NoOp => None,
        }
    }
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
        // `TraceRow` is `Copy` only without `field-inline` (which adds a non-`Copy` `Arc`
        // field), so the row is copied or cloned to match the active build.
        #[cfg(not(feature = "field-inline"))]
        let row = self.rows.get(self.next).copied();
        #[cfg(feature = "field-inline")]
        let row = self.rows.get(self.next).cloned();
        self.next += usize::from(row.is_some());
        row
    }

    fn rows(&self) -> Option<&[TraceRow]> {
        Some(OwnedTrace::rows(self))
    }
}
