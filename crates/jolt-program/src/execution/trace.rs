use std::{num::NonZeroU64, sync::Arc};

use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_riscv::{
    CircuitFlags, JoltInstructionProfile, JoltInstructionRow, JoltTraceRow, RV64IMAC_JOLT,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{ExecutionBackend, TraceError};

mod data;

#[cfg(feature = "field-inline")]
pub use data::FieldEvent;
pub use data::TraceData;

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
    bounds: Option<(NonZeroU64, NonZeroU64)>,
}

impl RamAddressBounds {
    /// Accept externally collected bounds if the minimum does not exceed the
    /// maximum. This does not validate against a program's memory layout.
    pub fn try_new(bounds: Option<(NonZeroU64, NonZeroU64)>) -> Result<Self, TraceError> {
        if bounds.is_some_and(|(min, max)| min > max) {
            return Err(TraceError::InvalidRamBounds);
        }
        Ok(Self { bounds })
    }

    pub fn observe(&mut self, address: u64) {
        if let Some(address) = NonZeroU64::new(address) {
            let (min, max) = self.bounds.get_or_insert((address, address));
            *min = (*min).min(address);
            *max = (*max).max(address);
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.bounds = match (self.bounds, other.bounds) {
            (Some((min, max)), Some((other_min, other_max))) => {
                Some((min.min(other_min), max.max(other_max)))
            }
            (bounds, None) | (None, bounds) => bounds,
        };
        self
    }

    pub fn min(&self) -> Option<NonZeroU64> {
        self.bounds.map(|(min, _)| min)
    }

    pub fn max(&self) -> Option<NonZeroU64> {
        self.bounds.map(|(_, max)| max)
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
    pub fn from_rows(rows: &[JoltTraceRow]) -> Self {
        let observe = |mut bounds: RamAddressBounds, row: &JoltTraceRow| {
            bounds.observe(row.ram_address());
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
            ends_in_jump: rows
                .last()
                .is_some_and(|row| row.circuit_flags().get(CircuitFlags::Jump)),
            ram_bounds,
        }
    }
}

/// [`ExecutionDimensions`] folded over contiguous row ranges that are observed
/// and merged in any order. The proof prefix follows [`TraceData::new`]: it
/// ends at the last row that is not canonical padding. Field-inline events
/// are not folded; field-inline traces take their prefix from
/// `TraceData::from_parts`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DimensionFold {
    /// Execution index and jump flag of the last non-padding row observed.
    last_row: Option<(usize, bool)>,
    ram_bounds: RamAddressBounds,
}

impl DimensionFold {
    /// Observe `rows`, the execution rows starting at index `first`.
    pub fn observe(&mut self, first: usize, rows: &[JoltTraceRow]) {
        for row in rows {
            self.ram_bounds.observe(row.ram_address());
        }
        let prefix = data::proof_prefix_len(rows);
        if let Some(row) = prefix.checked_sub(1).and_then(|last| rows.get(last)) {
            self.keep_last((
                first + prefix - 1,
                row.circuit_flags().get(CircuitFlags::Jump),
            ));
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        if let Some(row) = other.last_row {
            self.keep_last(row);
        }
        self.ram_bounds = self.ram_bounds.merge(other.ram_bounds);
        self
    }

    pub fn dimensions(self) -> ExecutionDimensions {
        ExecutionDimensions {
            trace_length: self.last_row.map_or(0, |(index, _)| index + 1),
            ends_in_jump: self.last_row.is_some_and(|(_, jump)| jump),
            ram_bounds: self.ram_bounds,
        }
    }

    fn keep_last(&mut self, row: (usize, bool)) {
        if self.last_row.is_none_or(|(index, _)| index < row.0) {
            self.last_row = Some(row);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraceOutput<T = Arc<TraceData>> {
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

impl TraceOutput<Arc<TraceData>> {
    /// Summarize retained rows when the producer has not already collected dimensions.
    pub fn new(
        trace: TraceData,
        device: JoltDevice,
        final_memory: Option<MemoryImage>,
        advice_tape: Option<Vec<u8>>,
    ) -> Self {
        Self::with_dimensions(
            ExecutionDimensions::from_rows(trace.proof_rows()),
            Arc::new(trace),
            device,
            final_memory,
            advice_tape,
        )
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test module")]
mod dimension_tests {
    use super::*;
    use jolt_riscv::{
        JoltInstructionKind, NormalizedOperands, RamAccess, RamRead, RamWrite, RegisterRead,
        RegisterState, RegisterWrite,
    };

    #[test]
    fn external_ram_bounds_must_be_ordered() {
        for (min, max) in [(16, 8), (u64::MAX, 1)] {
            assert!(matches!(
                RamAddressBounds::try_new(Some((
                    NonZeroU64::new(min).expect("nonzero minimum"),
                    NonZeroU64::new(max).expect("nonzero maximum"),
                ))),
                Err(TraceError::InvalidRamBounds)
            ));
        }
        for pair in [
            None,
            Some((8, 8)),
            Some((8, 16)),
            Some((u64::MAX, u64::MAX)),
        ] {
            let bounds = RamAddressBounds::try_new(pair.map(|(min, max)| {
                (
                    NonZeroU64::new(min).expect("nonzero minimum"),
                    NonZeroU64::new(max).expect("nonzero maximum"),
                )
            }))
            .expect("valid bounds");
            assert_eq!(
                (
                    bounds.min().map(NonZeroU64::get),
                    bounds.max().map(NonZeroU64::get)
                ),
                (pair.map(|(min, _)| min), pair.map(|(_, max)| max))
            );
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
                assert_eq!(
                    (
                        merged.min().map(NonZeroU64::get),
                        merged.max().map(NonZeroU64::get)
                    ),
                    expected
                );
                assert_eq!(merged.merge(RamAddressBounds::default()), merged);
            }
        }
    }

    fn row(
        kind: JoltInstructionKind,
        operands: NormalizedOperands,
        registers: RegisterState,
        ram: RamAccess,
    ) -> JoltTraceRow {
        let instruction = JoltInstructionRow {
            instruction_kind: kind,
            address: 0x8000_0000,
            operands,
            ..Default::default()
        };
        JoltTraceRow::new(instruction, registers, ram, 1).expect("valid row")
    }

    /// Folding any partition of the rows, merged in either order, reproduces
    /// the trimmed proof view of `TraceData::new` summarized by `from_rows`.
    #[test]
    fn dimension_fold_matches_proof_rows_over_any_partition() {
        let load = row(
            JoltInstructionKind::LD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: None,
                rd: Some(11),
                imm: 8,
            },
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 10,
                    value: 0x1000,
                }),
                rs2: None,
                rd: Some(RegisterWrite {
                    register: 11,
                    pre_value: 5,
                    post_value: 7,
                }),
            },
            RamAccess::Read(RamRead {
                address: 0x1008,
                value: 7,
            }),
        );
        let store = row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(10),
                rs2: Some(12),
                rd: None,
                imm: -4,
            },
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 10,
                    value: 0x3000,
                }),
                rs2: Some(RegisterRead {
                    register: 12,
                    value: 0x1234,
                }),
                rd: None,
            },
            RamAccess::Write(RamWrite {
                address: 0x2ffc,
                pre_value: 0,
                post_value: 0x1234,
            }),
        );
        let jump = row(
            JoltInstructionKind::JAL,
            NormalizedOperands {
                rs1: None,
                rs2: None,
                rd: Some(1),
                imm: 8,
            },
            RegisterState {
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 0,
                    post_value: 0x8000_0004,
                }),
                ..Default::default()
            },
            RamAccess::NoOp,
        );
        assert!(jump.circuit_flags().get(CircuitFlags::Jump));
        let pad = JoltTraceRow::default();
        for rows in [
            vec![],
            vec![pad, pad],
            vec![load, pad, store, jump, pad, pad],
            vec![jump, pad, load, pad],
            vec![pad, store],
        ] {
            let expected =
                ExecutionDimensions::from_rows(TraceData::new(rows.clone()).proof_rows());
            for cuts in 0u32..1 << rows.len().saturating_sub(1) {
                let mut folds = Vec::new();
                let (mut first, mut rest) = (0, rows.as_slice());
                while !rest.is_empty() {
                    let mut len = 1;
                    while len < rest.len() && cuts >> (first + len - 1) & 1 == 0 {
                        len += 1;
                    }
                    let (chunk, tail) = rest.split_at(len);
                    let mut fold = DimensionFold::default();
                    fold.observe(first, chunk);
                    folds.push(fold);
                    first += len;
                    rest = tail;
                }
                let forward = folds
                    .iter()
                    .fold(DimensionFold::default(), |all, fold| all.merge(*fold));
                let backward = folds
                    .iter()
                    .rev()
                    .fold(DimensionFold::default(), |all, fold| all.merge(*fold));
                assert_eq!(forward.dimensions(), expected, "cuts {cuts:b}");
                assert_eq!(backward.dimensions(), expected, "cuts {cuts:b}");
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
