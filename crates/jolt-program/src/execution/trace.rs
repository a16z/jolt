use std::sync::Arc;

use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_riscv::{
    CircuitFlags, JoltInstructionProfile, JoltInstructionRow, JoltTraceRow, RV64IMAC_JOLT,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    pub profile: JoltInstructionProfile,
}

impl Default for JoltProgram {
    fn default() -> Self {
        Self::from_elf_bytes(Vec::new())
    }
}

impl JoltProgram {
    pub fn from_elf_bytes(elf_bytes: Vec<u8>) -> Self {
        Self::from_elf_bytes_with_profile(elf_bytes, RV64IMAC_JOLT)
    }

    /// [`Self::from_elf_bytes`] under an explicit instruction profile. The
    /// profile is load-bearing beyond decode: the field-inline witness plane
    /// fails closed on field-inline trace data unless the program declares a
    /// profile that enables field-inline, so such guests must be constructed
    /// through here.
    pub fn from_elf_bytes_with_profile(
        elf_bytes: Vec<u8>,
        profile: JoltInstructionProfile,
    ) -> Self {
        Self {
            elf_bytes,
            expanded_bytecode: Vec::new(),
            memory_init: Vec::new(),
            program_end: 0,
            entry_address: 0,
            profile,
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

/// Bounds of nonzero RAM byte addresses. Zero denotes no access; an empty
/// accumulator has no bounds. Partial summaries can be merged in any order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamAddressBounds {
    min: Option<u64>,
    max: Option<u64>,
}

impl RamAddressBounds {
    /// Accept externally collected bounds without allowing partial, zero, or
    /// reversed bounds. This does not validate against a program's memory layout.
    pub fn try_new(min: Option<u64>, max: Option<u64>) -> Result<Self, TraceError> {
        match (min, max) {
            (None, None) => Ok(Self::default()),
            (Some(min), Some(max)) if min != 0 && min <= max => Ok(Self {
                min: Some(min),
                max: Some(max),
            }),
            _ => Err(TraceError::InvalidRamBounds),
        }
    }

    pub fn observe(&mut self, address: u64) {
        if address != 0 {
            self.min = Some(self.min.unwrap_or(address).min(address));
            self.max = Some(self.max.unwrap_or(address).max(address));
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        if let Some(min) = other.min {
            self.observe(min);
        }
        if let Some(max) = other.max {
            self.observe(max);
        }
        self
    }

    pub fn min(&self) -> Option<u64> {
        self.min
    }

    pub fn max(&self) -> Option<u64> {
        self.max
    }
}

/// Execution facts independent of trace storage and commitment-scheme padding.
/// `trace_length` excludes padding added by a proving adapter. Addresses are
/// byte addresses excluding zero; both bounds are `None` when there is no
/// nonzero RAM access. The producer must cover every
/// read and write, including accesses that leave memory unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionDimensions {
    pub trace_length: usize,
    /// Whether the final unpadded row has the protocol jump flag.
    pub ends_in_jump: bool,
    pub ram_bounds: RamAddressBounds,
}

#[cfg(feature = "parallel")]
const PARALLEL_SUMMARIZE_MIN_ROWS: usize = 1 << 16;

impl ExecutionDimensions {
    /// Summarize existing rows. Execution backends can instead collect
    /// these facts while producing rows and return them with an opaque handle.
    pub fn from_rows(rows: &[TraceRow]) -> Self {
        Self::summarize(
            rows,
            rows.last()
                .is_some_and(|row| row.circuit_flags().get(CircuitFlags::Jump)),
            |row| match row.ram_access() {
                RamAccess::Read(read) => Some(read.address),
                RamAccess::Write(write) => Some(write.address),
                RamAccess::NoOp => None,
            },
        )
    }

    pub fn from_compact(rows: &[JoltTraceRow]) -> Self {
        Self::summarize(
            rows,
            rows.last()
                .is_some_and(|row| row.circuit_flags().get(CircuitFlags::Jump)),
            |row| (row.is_load() || row.is_store()).then(|| row.ram_address()),
        )
    }

    fn summarize<R: Sync>(
        rows: &[R],
        ends_in_jump: bool,
        ram_address: impl Fn(&R) -> Option<u64> + Sync,
    ) -> Self {
        let observe = |mut bounds: RamAddressBounds, row: &R| {
            bounds.observe(ram_address(row).unwrap_or(0));
            bounds
        };
        #[cfg(feature = "parallel")]
        let ram_bounds = if rows.len() >= PARALLEL_SUMMARIZE_MIN_ROWS {
            rows.par_iter()
                .fold(RamAddressBounds::default, observe)
                .reduce(RamAddressBounds::default, RamAddressBounds::merge)
        } else {
            rows.iter().fold(RamAddressBounds::default(), observe)
        };
        #[cfg(not(feature = "parallel"))]
        let ram_bounds = rows.iter().fold(RamAddressBounds::default(), observe);
        Self {
            trace_length: rows.len(),
            ends_in_jump,
            ram_bounds,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraceOutput<T> {
    pub dimensions: ExecutionDimensions,
    pub trace: T,
    pub device: JoltDevice,
    pub final_memory: Option<MemoryImage>,
    /// The populated runtime advice tape captured at guest termination
    /// (`None` when the backend produced no tape).
    pub advice_tape: Option<Vec<u8>>,
}

impl<T> TraceOutput<T> {
    /// Dimensions are explicit so rebuilding or padding an output preserves
    /// the execution facts without scanning or requiring trace rows.
    /// `advice_tape` is a required parameter so that a backend (or a
    /// rebuild of an existing output) cannot silently discard a populated
    /// tape — the seam this field exists to plug.
    pub fn with_dimensions(
        dimensions: ExecutionDimensions,
        trace: T,
        device: JoltDevice,
        final_memory: Option<MemoryImage>,
        advice_tape: Option<Vec<u8>>,
    ) -> Self {
        Self {
            dimensions,
            trace,
            device,
            final_memory,
            advice_tape,
        }
    }
}

impl TraceOutput<OwnedTrace> {
    /// Summarize an unpadded owned trace. Use `with_dimensions` when the
    /// execution dimensions are already known or the rows have been padded.
    pub fn new(
        trace: OwnedTrace,
        device: JoltDevice,
        final_memory: Option<MemoryImage>,
        advice_tape: Option<Vec<u8>>,
    ) -> Self {
        Self::with_dimensions(
            ExecutionDimensions::from_rows(trace.rows()),
            trace,
            device,
            final_memory,
            advice_tape,
        )
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

    fn shared_rows(&self) -> Option<Arc<Vec<TraceRow>>> {
        (self.next == 0).then(|| Arc::clone(&self.rows))
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test module")]
mod dimension_tests {
    use super::*;

    #[test]
    fn external_ram_bounds_must_be_complete_nonzero_and_ordered() {
        for (min, max) in [
            (None, Some(8)),
            (Some(8), None),
            (Some(0), Some(8)),
            (Some(8), Some(0)),
            (Some(0), Some(0)),
            (Some(16), Some(8)),
        ] {
            assert!(matches!(
                RamAddressBounds::try_new(min, max),
                Err(TraceError::InvalidRamBounds)
            ));
        }
        for (min, max) in [
            (None, None),
            (Some(8), Some(8)),
            (Some(8), Some(16)),
            (Some(u64::MAX), Some(u64::MAX)),
        ] {
            let bounds = RamAddressBounds::try_new(min, max).expect("valid bounds");
            assert_eq!((bounds.min(), bounds.max()), (min, max));
        }
    }

    #[test]
    fn ram_bounds_merge_empty_and_nonzero_partitions() {
        for (addresses, expected) in [
            ([0, 0, 0, 0, 0], (None, None)),
            (
                [0, u64::MAX, 0, u64::MAX, 0],
                (Some(u64::MAX), Some(u64::MAX)),
            ),
            ([0, 17, 0, 9, u64::MAX], (Some(9), Some(u64::MAX))),
        ] {
            for split in 0..=addresses.len() {
                let mut left = RamAddressBounds::default();
                let mut right = RamAddressBounds::default();
                for &address in addresses.iter().take(split) {
                    left.observe(address);
                }
                for &address in addresses.iter().skip(split) {
                    right.observe(address);
                }
                let merged = left.merge(right);
                assert_eq!(merged, right.merge(left));
                assert_eq!((merged.min(), merged.max()), expected);
                assert_eq!(merged.merge(RamAddressBounds::default()), merged);
            }
        }
    }

    #[test]
    fn execution_backend_can_return_an_opaque_trace() {
        struct OpaqueTrace(u64);
        struct Backend;
        impl ExecutionBackend for Backend {
            type Trace = OpaqueTrace;
            fn trace(
                &mut self,
                _: &JoltProgram,
                _: TraceInputs,
            ) -> Result<TraceOutput<Self::Trace>, TraceError> {
                Ok(TraceOutput::with_dimensions(
                    ExecutionDimensions {
                        trace_length: 17,
                        ..Default::default()
                    },
                    OpaqueTrace(42),
                    JoltDevice::default(),
                    None,
                    None,
                ))
            }
        }
        let output = JoltProgram::default()
            .trace_with(&mut Backend, TraceInputs::default())
            .expect("valid execution dimensions");
        assert_eq!(output.dimensions.trace_length, 17);
        assert_eq!(output.trace.0, 42);
    }
}
