//! Canonical 128-bit Solinas-field arithmetic on Metal.
//!
//! [`Fp128`] is the buffer ABI, not a host field implementation. Arithmetic is
//! performed by the shader specialized for `2^128 - C`; host callers supply
//! canonical values for the selected offset.

use std::{cell::Cell, ffi::c_void, slice, time::Duration};

use jolt_field::FixedBytes;
use metal::{
    objc::{rc::autoreleasepool, runtime::Sel, Message},
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;

const FIELD_SOURCE: &str = include_str!("fp128.metal");
const ADDRESS_RAF_SOURCE: &str = include_str!("address_raf.metal");
const ADDRESS_RAF_DIRECT_SOURCE: &str = include_str!("address_raf_direct.metal");
const ADDRESS_SUFFIX_SOURCE: &str = include_str!("address_suffix.metal");
const ADDRESS_SUFFIX_FULL_SOURCE: &str = include_str!("address_suffix_full.metal");
const ADDRESS_CYCLE_SOURCE: &str = include_str!("address_cycle.metal");
const PROBE_SOURCE: &str = include_str!("probes.metal");
const PRODUCT5_SOURCE: &str = include_str!("product5.metal");
const BOOLEANITY_SOURCE: &str = include_str!("booleanity.metal");
const INSTRUCTION_RA_SOURCE: &str = include_str!("instruction_ra_virtualization.metal");
const INSTRUCTION_RA_SEQUENCE_SOURCE: &str = include_str!("instruction_ra_sequence.metal");
const INSTRUCTION_INPUT_SOURCE: &str = include_str!("instruction_input.metal");
const BYTECODE_CYCLE_SOURCE: &str = include_str!("bytecode_cycle.metal");
const BYTECODE_ROW_SOURCE: &str = include_str!("bytecode_row.metal");
const SPARTAN_OUTER_UNISKIP_SOURCE: &str = include_str!("spartan_outer_uniskip.metal");

mod address_raf;
mod address_raf_direct;
mod address_sequence;
mod address_suffix;
mod address_suffix_full;
mod booleanity;
mod bytecode_cycle;
mod bytecode_row;
mod instruction_input;
mod instruction_ra_sequence;
mod instruction_ra_virtualization;
mod product5;
mod spartan_outer_uniskip;

pub use address_raf::{
    AddressRafScanConfig, AddressRafScanInvocation, AddressRafScanRow, AddressRafSums,
    ADDRESS_RAF_BINS, ADDRESS_RAF_LANES,
};
pub use address_raf_direct::AddressRafDirectInvocation;
pub(crate) use address_sequence::ResidentLookupIndexPlane;
pub use address_sequence::{AddressPhaseSequence, AddressPhaseSequenceConfig, AddressPhaseSums};
pub use address_suffix::{
    AddressSuffixOneInvocation, AddressSuffixOneSums, ADDRESS_SUFFIX_BINS, ADDRESS_SUFFIX_TABLES,
};
pub use address_suffix_full::{AddressSuffixFullInvocation, AddressSuffixFullSums};
pub(crate) use booleanity::BooleanityRows;
pub use booleanity::{
    BooleanityRow, BooleanitySelector, BooleanitySequence, BooleanitySequenceConfig,
};
pub use bytecode_cycle::{
    BytecodeCycleSequence, BytecodeCycleSequenceConfig, BytecodeCycleTables,
    BytecodeCycleTablesMut, BYTECODE_CYCLE_SAMPLES, BYTECODE_CYCLE_TABLES,
};
pub(crate) use bytecode_row::{BytecodeCycleRowInputs, BytecodeCycleRowSequence};
pub(crate) use instruction_input::{
    instruction_input_sequence_storage_bytes, instruction_input_weight_capacities,
    InstructionInputSequenceStorage,
};
pub use instruction_input::{
    InstructionInputSequence, InstructionInputSequenceConfig, INSTRUCTION_INPUT_COEFFICIENTS,
    INSTRUCTION_INPUT_TABLES,
};
pub(crate) use instruction_ra_sequence::{
    instruction_ra_weight_capacities, InstructionRaSequenceStorage,
};
pub use instruction_ra_sequence::{
    InstructionRaLookupPlane, InstructionRaMaterializeWidth, InstructionRaSequence,
    InstructionRaSequenceConfig, InstructionRaSequenceScratchLayout,
};
pub use instruction_ra_virtualization::{
    InstructionRaFirstMessageConfig, InstructionRaFirstMessageInvocation,
};
pub use product5::{
    Product5Config, Product5Invocation, Product5Sequence, Product5SequenceConfig, PRODUCT5_FACTORS,
};
pub use spartan_outer_uniskip::{
    evaluate_spartan_outer_uniskip_cpu, SpartanOuterUniskipConfig, SpartanOuterUniskipInvocation,
    SpartanOuterUniskipRow, SpartanOuterUniskipRows, SPARTAN_OUTER_EXTENDED_NODES,
};
pub(crate) use spartan_outer_uniskip::{
    spartan_outer_uniskip_invocation_bytes, spartan_outer_uniskip_row_bytes,
};

pub const OFFSET_275: u32 = 275;
pub const AKITA_OFFSET_FFFFA7F7: u32 = 0xffff_a7f7;

/// Little-endian limbs shared by Rust and Metal buffers.
///
/// Dispatch validates canonicality for the selected Solinas offset.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp128 {
    limbs: [u32; 4],
}

impl Fp128 {
    pub const ZERO: Self = Self::from_u128(0);
    pub const ONE: Self = Self::from_u128(1);

    pub const fn from_limbs(limbs: [u32; 4]) -> Self {
        Self { limbs }
    }

    pub const fn from_u128(value: u128) -> Self {
        Self {
            limbs: [
                value as u32,
                (value >> 32) as u32,
                (value >> 64) as u32,
                (value >> 96) as u32,
            ],
        }
    }

    pub const fn limbs(self) -> [u32; 4] {
        self.limbs
    }

    pub const fn to_u128(self) -> u128 {
        (self.limbs[0] as u128)
            | ((self.limbs[1] as u128) << 32)
            | ((self.limbs[2] as u128) << 64)
            | ((self.limbs[3] as u128) << 96)
    }

    pub const fn is_canonical(self, offset: u32) -> bool {
        offset != 0 && self.to_u128() <= u128::MAX - offset as u128
    }

    pub fn from_jolt_field<F: FixedBytes<16>>(value: &F) -> Self {
        Self::from_u128(u128::from_le_bytes(value.to_bytes_array()))
    }

    pub fn into_jolt_field<F: FixedBytes<16>>(self) -> F {
        F::from_bytes_array(&self.to_u128().to_le_bytes())
    }
}

/// A compiled entry point used to characterize one part of the field pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Probe {
    Noop,
    Copy,
    Add,
    Sub,
    MulWide,
    ChainWide1,
    ChainWide2,
    ChainWide4,
    ChainWide8,
    U32MadIlp8,
}

impl Probe {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Noop => "solinas_noop",
            Self::Copy => "solinas_copy",
            Self::Add => "solinas_add_probe",
            Self::Sub => "solinas_sub_probe",
            Self::MulWide => "solinas_mul_wide_probe",
            Self::ChainWide1 => "solinas_chain_wide_1",
            Self::ChainWide2 => "solinas_chain_wide_2",
            Self::ChainWide4 => "solinas_chain_wide_4",
            Self::ChainWide8 => "solinas_chain_wide_8",
            Self::U32MadIlp8 => "solinas_u32_mad_ilp8",
        }
    }

    pub const fn independent_chains(self) -> usize {
        match self {
            Self::ChainWide2 => 2,
            Self::ChainWide4 => 4,
            Self::ChainWide8 => 8,
            _ => 1,
        }
    }

    const fn accepts_noncanonical_output(self) -> bool {
        matches!(self, Self::U32MadIlp8)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchConfig {
    pub iterations: u32,
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            threads_per_threadgroup: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineLimits {
    pub thread_execution_width: usize,
    pub max_total_threads_per_threadgroup: usize,
    pub static_threadgroup_memory_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub max_buffer_length: u64,
    pub max_threadgroup_memory_length: u64,
    pub recommended_max_working_set_size: u64,
    pub current_allocated_size: u64,
    pub offset: u32,
}

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("no Metal device is available")]
    DeviceUnavailable,
    #[error("Solinas offset must be nonzero")]
    InvalidOffset,
    #[error("kernel requires Solinas offset {expected:#x}, but the context uses {got:#x}")]
    UnexpectedSolinasOffset { expected: u32, got: u32 },
    #[error("failed to compile the Solinas Metal library: {0}")]
    LibraryCompilation(String),
    #[error("Metal entry point `{name}` was not found: {message}")]
    FunctionLookup { name: &'static str, message: String },
    #[error("failed to compile Metal entry point `{name}`: {message}")]
    PipelineCompilation { name: &'static str, message: String },
    #[error("a non-noop dispatch requires at least one element")]
    EmptyInput,
    #[error("use `prepare_noop` for the no-op probe")]
    NoopPreparation,
    #[error("input lengths differ: lhs={lhs}, rhs={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("input length {0} exceeds the shader's 32-bit element count")]
    InputTooLong(usize),
    #[error("buffer requires {requested} bytes but the Metal device limit is {maximum}")]
    BufferTooLong { requested: u64, maximum: u64 },
    #[error(
        "Metal has {current} bytes allocated and the kernel needs {additional} more, exceeding the recommended working set of {maximum} bytes"
    )]
    WorkingSetTooLarge {
        current: u64,
        additional: u64,
        maximum: u64,
    },
    #[error("input {side}[{index}] is not canonical for 2^128 - {offset}")]
    NonCanonicalInput {
        side: &'static str,
        index: usize,
        offset: u32,
    },
    #[error("output[{index}] is not canonical for 2^128 - {offset}")]
    NonCanonicalOutput { index: usize, offset: u32 },
    #[error("{probe} requires an element count divisible by its ILP ({ilp})")]
    MisalignedElementCount { probe: &'static str, ilp: usize },
    #[error("iteration count must be nonzero")]
    ZeroIterations,
    #[error("Spartan outer uni-skip shape mismatch: rows={rows}, e_in={e_in}, e_out={e_out}")]
    SpartanOuterUniskipShape {
        rows: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error(
        "Spartan outer uni-skip pipeline `{pipeline}` has execution width {got}, expected {expected}"
    )]
    UnsupportedSpartanOuterExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "Spartan outer uni-skip needs {requested} bytes of threadgroup memory, device limit is {maximum}"
    )]
    SpartanOuterThreadgroupMemory { requested: u64, maximum: u64 },
    #[error("Spartan outer row {row} could not be extracted: {message}")]
    SpartanOuterRowExtraction { row: usize, message: String },
    #[error("address RAF row and weight lengths differ: rows={rows}, weights={weights}")]
    AddressRafLengthMismatch { rows: usize, weights: usize },
    #[error("address RAF suffix length must be a multiple of eight in 0..=120, got {0}")]
    InvalidAddressRafSuffixLength(u32),
    #[error("address RAF condensation requires a suffix length in 0..=112, got {0}")]
    InvalidAddressRafCondensationSuffixLength(u32),
    #[error("address RAF rows per threadgroup must be nonzero, got {0}")]
    InvalidAddressRafRowsPerThreadgroup(usize),
    #[error("direct address RAF rows per threadgroup must be in 1..=65536, got {0}")]
    InvalidAddressRafDirectRowsPerThreadgroup(usize),
    #[error(
        "direct address RAF needs {requested} bytes of threadgroup memory, device limit is {maximum}"
    )]
    AddressRafDirectThreadgroupMemory { requested: u64, maximum: u64 },
    #[error("address suffix row selects unknown table {0}")]
    InvalidAddressSuffixTable(usize),
    #[error("address suffix scan requires at least one table-selected row")]
    EmptyAddressSuffixBuckets,
    #[error("address phase needs {expected} table buckets, got {got}")]
    AddressPhaseBucketCount { expected: usize, got: usize },
    #[error("address phase bucket {bucket} contains row {row} for table {actual:?}")]
    InvalidAddressPhaseBucket {
        bucket: usize,
        row: usize,
        actual: Option<usize>,
    },
    #[error("address phase table-major layout has {got} rows, expected {expected}")]
    AddressPhaseLayoutLength { expected: usize, got: usize },
    #[error("address cycle phase tables contain {got} fields, expected {expected}")]
    AddressCyclePhaseTableShape { expected: usize, got: usize },
    #[error("address cycle has {got} table values, expected {expected}")]
    AddressCycleTableValueCount { expected: usize, got: usize },
    #[error("lookup table {table} has {count} suffixes; Metal supports at most {maximum}")]
    InvalidAddressSuffixCount {
        table: usize,
        count: usize,
        maximum: usize,
    },
    #[error(
        "address suffix kernel needs {requested} bytes of threadgroup memory, device limit is {maximum}"
    )]
    AddressSuffixThreadgroupMemory { requested: u64, maximum: u64 },
    #[error(
        "address RAF pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedAddressRafExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "address cycle pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedAddressCycleExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("hybrid cutoff must be a power of two of at least two, got {0}")]
    InvalidHybridCutoff(usize),
    #[error(
        "Instruction RA cutoff {instruction_ra_cutoff} is below the address-plane cutoff {address_cutoff}"
    )]
    InstructionRaRequiresAddressPlane {
        instruction_ra_cutoff: usize,
        address_cutoff: usize,
    },
    #[error("Instruction RA needs a power-of-two row count of at least two, got {0}")]
    InvalidInstructionRaRows(usize),
    #[error("Instruction RA factor-table storage has length {got}, expected {expected}")]
    InstructionRaStorageLength { expected: usize, got: usize },
    #[error("Instruction RA split weights cover {covered} pairs, expected {expected}")]
    InstructionRaWeightShape { expected: usize, covered: usize },
    #[error("Instruction RA resident {name} buffer has {got} bytes, expected {expected}")]
    InstructionRaPlaneLength {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    #[error(
        "Instruction RA resident plane belongs to Metal device {got}, but the kernel uses {expected}"
    )]
    InstructionRaPlaneDevice { expected: u64, got: u64 },
    #[error(
        "Instruction RA pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedInstructionRaExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid resident Instruction RA state: {0}")]
    InvalidInstructionRaState(&'static str),
    #[error("InstructionInput needs a power-of-two row count of at least four, got {0}")]
    InvalidInstructionInputRows(usize),
    #[error("InstructionInput table storage has length {got}, expected {expected}")]
    InstructionInputStorageLength { expected: usize, got: usize },
    #[error("InstructionInput split weights cover {covered} pairs, expected {expected}")]
    InstructionInputWeightShape { expected: usize, covered: usize },
    #[error("InstructionInput rows belong to Metal device {got}, but the kernel uses {expected}")]
    InstructionInputRowsDevice { expected: u64, got: u64 },
    #[error(
        "InstructionInput pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedInstructionInputExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid resident InstructionInput state: {0}")]
    InvalidInstructionInputState(&'static str),
    #[error(
        "bytecode cycle kernels require a power-of-two table length of at least {minimum}, got {got}"
    )]
    InvalidBytecodeCycleTableLength { minimum: usize, got: usize },
    #[error("bytecode cycle plane `{plane}` has length {got}, expected {expected}")]
    BytecodeCyclePlaneLength {
        plane: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("bytecode cycle maximum threadgroup count must be nonzero, got {0}")]
    InvalidBytecodeCycleThreadgroups(usize),
    #[error("invalid resident bytecode cycle state: {0}")]
    InvalidBytecodeCycleState(&'static str),
    #[error(
        "bytecode cycle pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedBytecodeCycleExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "row-derived bytecode cycle needs {expected} stages, got {points} points and {weights} weights"
    )]
    BytecodeCycleRowStageCount {
        expected: usize,
        points: usize,
        weights: usize,
    },
    #[error("row-derived bytecode cycle point {stage} has length {got}, expected {expected}")]
    BytecodeCycleRowPointLength {
        stage: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "row-derived bytecode cycle needs {required} threadgroups, configured maximum is {maximum}"
    )]
    BytecodeCycleRowThreadgroups { required: usize, maximum: usize },
    #[error(
        "five-factor kernels require a power-of-two table length of at least {minimum}, got {got}"
    )]
    InvalidProduct5TableLength { minimum: usize, got: usize },
    #[error("five-factor table storage has length {got}, expected {expected}")]
    Product5StorageLength { expected: usize, got: usize },
    #[error(
        "split equality tables cover {covered} pairs, but the five-factor kernel needs {expected}"
    )]
    Product5WeightShape { expected: usize, covered: usize },
    #[error(
        "five-factor pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedProduct5ExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("booleanity row cannot be represented by the packed Metal ABI")]
    InvalidBooleanityRow,
    #[error("booleanity needs a power-of-two row count of at least four, got {0}")]
    InvalidBooleanityRows(usize),
    #[error("booleanity chunk size must be a power of two in 2..=256, got {0}")]
    InvalidBooleanityK(usize),
    #[error("booleanity materialization width must be a power of two in 1..=32, got {0}")]
    InvalidBooleanityMaterializeWidth(usize),
    #[error("booleanity selector is outside its packed source")]
    InvalidBooleanitySelector,
    #[error("booleanity {name} storage has length {got}, expected {expected}")]
    BooleanityStorageLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("Booleanity rows belong to Metal device {got}, but the kernel uses {expected}")]
    BooleanityRowsDevice { expected: u64, got: u64 },
    #[error("booleanity split weights cover {covered} pairs, expected {expected}")]
    BooleanityWeightShape { expected: usize, covered: usize },
    #[error(
        "booleanity pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedBooleanityExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid booleanity sequence state: {0}")]
    InvalidBooleanityState(&'static str),
    #[error(
        "threadgroup width {requested} must be a multiple of {execution_width} and at most {maximum}"
    )]
    InvalidThreadgroupWidth {
        requested: usize,
        execution_width: usize,
        maximum: usize,
    },
    #[error("Metal command buffer finished with status {0:?}")]
    CommandFailed(MTLCommandBufferStatus),
    #[error("failed to read Metal command-buffer timestamp `{name}`: {message}")]
    GpuTimestampLookup { name: &'static str, message: String },
    #[error("Metal returned invalid GPU timestamps: start={start}, end={end}")]
    InvalidGpuTimestamps { start: f64, end: f64 },
    #[error("execute the invocation before reading its output")]
    NotExecuted,
}

#[derive(Clone)]
pub struct SolinasMetal {
    device: Device,
    queue: CommandQueue,
    library: Library,
    offset: u32,
}

impl SolinasMetal {
    pub fn for_akita() -> Result<Self, MetalError> {
        Self::new(AKITA_OFFSET_FFFFA7F7)
    }

    pub fn for_offset_275() -> Result<Self, MetalError> {
        Self::new(OFFSET_275)
    }

    pub(crate) fn device_registry_id(&self) -> u64 {
        self.device.registry_id()
    }

    pub fn new(offset: u32) -> Result<Self, MetalError> {
        if offset == 0 {
            return Err(MetalError::InvalidOffset);
        }
        let device = Device::system_default().ok_or(MetalError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let source = format!(
            "#define SOLINAS_OFFSET {offset}u\n{FIELD_SOURCE}\n{ADDRESS_RAF_SOURCE}\n{ADDRESS_RAF_DIRECT_SOURCE}\n{ADDRESS_SUFFIX_SOURCE}\n{ADDRESS_SUFFIX_FULL_SOURCE}\n{PROBE_SOURCE}\n{PRODUCT5_SOURCE}\n{BOOLEANITY_SOURCE}\n{INSTRUCTION_RA_SOURCE}\n{INSTRUCTION_RA_SEQUENCE_SOURCE}\n{BYTECODE_CYCLE_SOURCE}\n{BYTECODE_ROW_SOURCE}\n{SPARTAN_OUTER_UNISKIP_SOURCE}\n{INSTRUCTION_INPUT_SOURCE}\n{ADDRESS_CYCLE_SOURCE}"
        );
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(MetalError::LibraryCompilation)?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            library,
            offset,
        })
    }

    pub fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name().to_owned(),
            max_buffer_length: self.device.max_buffer_length(),
            max_threadgroup_memory_length: self.device.max_threadgroup_memory_length(),
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            current_allocated_size: self.device.current_allocated_size(),
            offset: self.offset,
        }
    }

    pub(crate) fn validate_additional_working_set(
        &self,
        additional: u64,
    ) -> Result<(), MetalError> {
        let current = self.device.current_allocated_size();
        let maximum = self.device.recommended_max_working_set_size();
        validate_working_set(current, additional, maximum)
    }

    pub fn pipeline_limits(&self, probe: Probe) -> Result<PipelineLimits, MetalError> {
        let pipeline = self.compile_pipeline(probe)?;
        Ok(Self::limits(&pipeline))
    }

    pub fn prepare_noop(&self) -> Result<Invocation<'_>, MetalError> {
        let pipeline = self.compile_pipeline(Probe::Noop)?;
        let limits = Self::limits(&pipeline);
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(Some(limits.thread_execution_width), limits)?;

        Ok(Invocation {
            context: self,
            probe: Probe::Noop,
            pipeline,
            buffers: None,
            limits,
            threads_per_threadgroup,
            grid_threads: 1,
            elements: 0,
            iterations: 1,
            completed: Cell::new(false),
        })
    }

    pub fn prepare(
        &self,
        probe: Probe,
        lhs: &[Fp128],
        rhs: &[Fp128],
        config: DispatchConfig,
    ) -> Result<Invocation<'_>, MetalError> {
        if probe == Probe::Noop {
            return Err(MetalError::NoopPreparation);
        }
        if lhs.is_empty() {
            return Err(MetalError::EmptyInput);
        }
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.len(),
            });
        }
        if config.iterations == 0 {
            return Err(MetalError::ZeroIterations);
        }
        let elements = u32::try_from(lhs.len()).map_err(|_| MetalError::InputTooLong(lhs.len()))?;
        let buffer_bytes =
            u64::try_from(size_of_val(lhs)).map_err(|_| MetalError::InputTooLong(lhs.len()))?;
        let max_buffer_length = self.device.max_buffer_length();
        if buffer_bytes > max_buffer_length {
            return Err(MetalError::BufferTooLong {
                requested: buffer_bytes,
                maximum: max_buffer_length,
            });
        }
        self.validate_inputs("lhs", lhs)?;
        self.validate_inputs("rhs", rhs)?;

        let ilp = probe.independent_chains();
        if !lhs.len().is_multiple_of(ilp) {
            return Err(MetalError::MisalignedElementCount {
                probe: probe.name(),
                ilp,
            });
        }

        let pipeline = self.compile_pipeline(probe)?;
        let limits = Self::limits(&pipeline);
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, limits)?;
        let grid_threads = lhs.len() / ilp;
        let params = ProbeParams {
            elements,
            iterations: config.iterations,
        };
        let buffers = Buffers {
            lhs: buffer_from_slice(&self.device, lhs),
            rhs: buffer_from_slice(&self.device, rhs),
            output: self
                .device
                .new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared),
            params: buffer_from_slice(&self.device, slice::from_ref(&params)),
        };

        Ok(Invocation {
            context: self,
            probe,
            pipeline,
            buffers: Some(buffers),
            limits,
            threads_per_threadgroup,
            grid_threads,
            elements: lhs.len(),
            iterations: config.iterations,
            completed: Cell::new(false),
        })
    }

    fn compile_pipeline(&self, probe: Probe) -> Result<ComputePipelineState, MetalError> {
        self.compile_named_pipeline(probe.name())
    }

    fn compile_named_pipeline(
        &self,
        name: &'static str,
    ) -> Result<ComputePipelineState, MetalError> {
        let function = self
            .library
            .get_function(name, None)
            .map_err(|message| MetalError::FunctionLookup { name, message })?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation { name, message })
    }

    fn validate_inputs(&self, side: &'static str, values: &[Fp128]) -> Result<(), MetalError> {
        #[cfg(feature = "parallel")]
        let invalid = values
            .par_iter()
            .enumerate()
            .find_first(|(_, value)| !value.is_canonical(self.offset));
        #[cfg(not(feature = "parallel"))]
        let invalid = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_canonical(self.offset));
        if let Some((index, _)) = invalid {
            return Err(MetalError::NonCanonicalInput {
                side,
                index,
                offset: self.offset,
            });
        }
        Ok(())
    }

    fn limits(pipeline: &ComputePipelineState) -> PipelineLimits {
        PipelineLimits {
            thread_execution_width: pipeline.thread_execution_width() as usize,
            max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup()
                as usize,
            static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
        }
    }

    fn resolve_threadgroup_width(
        requested: Option<usize>,
        limits: PipelineLimits,
    ) -> Result<usize, MetalError> {
        let execution_width = limits.thread_execution_width;
        let maximum = limits.max_total_threads_per_threadgroup;
        let default = (execution_width * 8).min(maximum);
        let width = requested.unwrap_or(default);
        if width == 0 || width > maximum || !width.is_multiple_of(execution_width) {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: width,
                execution_width,
                maximum,
            });
        }
        Ok(width)
    }
}

pub(crate) fn validate_working_set(
    current: u64,
    additional: u64,
    maximum: u64,
) -> Result<(), MetalError> {
    if current
        .checked_add(additional)
        .is_none_or(|total| total > maximum)
    {
        return Err(MetalError::WorkingSetTooLarge {
            current,
            additional,
            maximum,
        });
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ProbeParams {
    elements: u32,
    iterations: u32,
}

struct Buffers {
    lhs: Buffer,
    rhs: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct Invocation<'a> {
    context: &'a SolinasMetal,
    probe: Probe,
    pipeline: ComputePipelineState,
    buffers: Option<Buffers>,
    limits: PipelineLimits,
    threads_per_threadgroup: usize,
    grid_threads: usize,
    elements: usize,
    iterations: u32,
    completed: Cell<bool>,
}

impl Invocation<'_> {
    pub const fn probe(&self) -> Probe {
        self.probe
    }

    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn grid_threads(&self) -> usize {
        self.grid_threads
    }

    pub const fn iterations(&self) -> u32 {
        self.iterations
    }

    pub const fn field_operation_count(&self) -> u64 {
        match self.probe {
            Probe::Add | Probe::Sub | Probe::MulWide => self.elements as u64,
            Probe::ChainWide1 | Probe::ChainWide2 | Probe::ChainWide4 | Probe::ChainWide8 => {
                self.elements as u64 * self.iterations as u64
            }
            _ => 0,
        }
    }

    pub const fn logical_bytes(&self) -> u64 {
        let bytes_per_element = match self.probe {
            Probe::Copy => 32,
            Probe::Add
            | Probe::Sub
            | Probe::MulWide
            | Probe::ChainWide1
            | Probe::ChainWide2
            | Probe::ChainWide4
            | Probe::ChainWide8
            | Probe::U32MadIlp8 => 48,
            Probe::Noop => 0,
        };
        self.elements as u64 * bytes_per_element
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    /// Executes the command and returns time spent running on the GPU.
    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            if let Some(buffers) = &self.buffers {
                encoder.set_buffer(0, Some(&buffers.lhs), 0);
                encoder.set_buffer(1, Some(&buffers.rhs), 0);
                encoder.set_buffer(2, Some(&buffers.output), 0);
                encoder.set_buffer(3, Some(&buffers.params), 0);
            }
            let threads_per_threadgroup = MTLSize {
                width: self.threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = MTLSize {
                width: self.grid_threads.div_ceil(self.threads_per_threadgroup) as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
    }

    pub fn read_output(&self) -> Result<Vec<Fp128>, MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let Some(buffers) = &self.buffers else {
            return Ok(Vec::new());
        };
        // SAFETY: `output` is shared storage allocated for exactly `elements`
        // `Fp128` values and GPU execution is complete before callers read it.
        let output = unsafe {
            slice::from_raw_parts(buffers.output.contents().cast::<Fp128>(), self.elements).to_vec()
        };
        if !self.probe.accepts_noncanonical_output() {
            if let Some((index, _)) = output
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_canonical(self.context.offset))
            {
                return Err(MetalError::NonCanonicalOutput {
                    index,
                    offset: self.context.offset,
                });
            }
        }
        Ok(output)
    }
}

fn buffer_from_slice<T>(device: &Device, values: &[T]) -> Buffer {
    debug_assert!(!values.is_empty());
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn command_buffer_timestamp(
    command_buffer: &metal::CommandBufferRef,
    name: &'static str,
) -> Result<f64, MetalError> {
    // SAFETY: both selectors are required, argument-free MTLCommandBuffer
    // properties returning CFTimeInterval, which is an f64.
    unsafe { command_buffer.send_message::<(), f64>(Sel::register(name), ()) }.map_err(|error| {
        MetalError::GpuTimestampLookup {
            name,
            message: error.to_string(),
        }
    })
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test module")]
mod tests {
    use std::mem::{align_of, size_of};

    use super::{
        validate_working_set, AddressRafScanConfig, AddressRafScanRow, Fp128, MetalError,
        SolinasMetal, OFFSET_275,
    };

    #[test]
    fn working_set_admission_is_exact_and_overflow_safe() {
        assert!(validate_working_set(40, 60, 100).is_ok());
        assert!(matches!(
            validate_working_set(40, 61, 100),
            Err(MetalError::WorkingSetTooLarge {
                current: 40,
                additional: 61,
                maximum: 100,
            })
        ));
        assert!(matches!(
            validate_working_set(u64::MAX, 1, u64::MAX),
            Err(MetalError::WorkingSetTooLarge { .. })
        ));
    }

    #[test]
    fn fp128_has_the_metal_buffer_layout() {
        assert_eq!(size_of::<Fp128>(), 16);
        assert_eq!(align_of::<Fp128>(), 16);
    }

    #[test]
    fn limbs_are_little_endian() {
        let value = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
        let encoded = Fp128::from_u128(value);

        assert_eq!(encoded.to_u128(), value);
        assert_eq!(
            encoded.limbs(),
            [0x7654_3210, 0xfedc_ba98, 0x89ab_cdef, 0x0123_4567]
        );
    }

    #[test]
    fn canonicality_uses_the_selected_offset() {
        let largest = Fp128::from_u128(u128::MAX - OFFSET_275 as u128);
        let modulus = Fp128::from_u128(u128::MAX - OFFSET_275 as u128 + 1);

        assert!(largest.is_canonical(OFFSET_275));
        assert!(!modulus.is_canonical(OFFSET_275));
        assert!(!Fp128::ZERO.is_canonical(0));
    }

    #[test]
    fn address_raf_scan_reduces_exact_field_bins() {
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let rows = vec![AddressRafScanRow::new(0, false); 64];
        let weights: Vec<Fp128> = (1..=64).map(Fp128::from_u128).collect();
        let invocation = context
            .prepare_address_raf_scan(
                &rows,
                &weights,
                AddressRafScanConfig {
                    suffix_len: 120,
                    ..AddressRafScanConfig::default()
                },
            )
            .expect("address RAF scan should prepare");
        assert_eq!(
            invocation.intermediate_contribution_bytes(),
            rows.len() as u64 * 32
        );

        invocation
            .execute()
            .expect("address RAF scan should execute");
        let sums = invocation
            .read_output()
            .expect("address RAF output should be readable");
        let expected = (1u128..=64).sum();
        assert_eq!(sums.shift_half()[0], Fp128::from_u128(expected));
        assert!(sums
            .as_flat_slice()
            .iter()
            .enumerate()
            .all(|(index, value)| index == 0 || *value == Fp128::ZERO));
    }
}
