//! Canonical 128-bit Solinas-field arithmetic on Metal.
//!
//! [`Fp128`] is the buffer ABI, not a host field implementation. Arithmetic is
//! performed by the shader specialized for `2^128 - C`; host callers supply
//! canonical values for the selected offset.

use jolt_field::{CanonicalBytes, CanonicalEncoding};
use metal::MTLCommandBufferStatus;
use thiserror::Error;

macro_rules! copy_field_getters {
    ($visibility:vis, { $($method:ident $(=> $field:ident)?: $type:ty),* $(,)? }) => {
        $(
            $visibility const fn $method(&self) -> $type {
                copy_field_getters!(@get self, $method $(=> $field)?)
            }
        )*
    };
    (@get $self:ident, $method:ident) => {
        $self.$method
    };
    (@get $self:ident, $method:ident => $field:ident) => {
        $self.$field
    };
}

macro_rules! ref_field_getters {
    ($visibility:vis, { $($method:ident $(=> $field:ident)?: $type:ty),* $(,)? }) => {
        $(
            $visibility fn $method(&self) -> &$type {
                ref_field_getters!(@get self, $method $(=> $field)?)
            }
        )*
    };
    (@get $self:ident, $method:ident) => {
        &$self.$method
    };
    (@get $self:ident, $method:ident => $field:ident) => {
        &$self.$field
    };
}

mod address_raf;
mod address_sequence;
mod address_suffix_full;
mod booleanity;
mod booleanity_address;
mod bytecode_cycle;
#[doc(hidden)]
pub mod bytecode_read_raf_address;
mod bytecode_row;
mod hang_watchdog;
pub mod instruction_claim_reduction;
mod instruction_claim_reduction_successor;
mod instruction_input;
mod instruction_ra_sequence;
mod instruction_read_raf;
mod outer_remainder;
mod product5;
mod product_remainder;
mod product_uniskip;
#[doc(hidden)]
pub mod ram_cycle_family;
mod ram_hamming_sequence;
mod ram_ra_claim_reduction;
mod ram_ra_sequence;
mod ram_raf_evaluation;
mod ram_read_write;
mod ram_val_sequence;
pub mod registers_claim_reduction;
pub(crate) mod registers_read_write;
mod registers_val;
mod runtime;
mod source;
mod spartan_outer_uniskip;
pub mod spartan_shift;

use runtime::{
    buffer_from_slice, completed_command_gpu_time, encode_column_reductions,
    validate_completed_command, ReductionBuffer,
};
pub(crate) use runtime::{set_inline_bytes, validate_working_set};
pub use runtime::{DeviceInfo, PipelineLimits, SolinasMetal};

pub use address_raf::{AddressRafScanRow, AddressRafSums, ADDRESS_RAF_BINS, ADDRESS_RAF_LANES};
pub(crate) use address_sequence::ResidentLookupIndexPlane;
pub use address_sequence::{AddressPhaseSequence, AddressPhaseSequenceConfig, AddressPhaseSums};
pub use address_suffix_full::{AddressSuffixFullSums, ADDRESS_SUFFIX_BINS, ADDRESS_SUFFIX_TABLES};
pub use booleanity::BooleanityRows;
pub use booleanity::{
    BooleanityRow, BooleanitySelector, BooleanitySequence, BooleanitySequenceConfig,
};
pub(crate) use booleanity::{BOOLEANITY_SOURCE_ROW_BYTES, BOOLEANITY_SOURCE_WORDS};
pub use booleanity_address::{BooleanityAddressPushforward, BooleanityAddressPushforwardConfig};
pub use bytecode_cycle::{
    BytecodeCycleSequence, BytecodeCycleSequenceConfig, BytecodeCycleTables,
    BytecodeCycleTablesMut, BYTECODE_CYCLE_SAMPLES, BYTECODE_CYCLE_TABLES,
};
pub(crate) use bytecode_row::{BytecodeCycleRowInputs, BytecodeCycleRowSequence};
pub(crate) use instruction_claim_reduction_successor::{
    PendingProductInstructionInitialMessage, ProductInstructionOpenings,
    ProductInstructionRoundService, ProductInstructionRoundStats,
};
pub(crate) use instruction_input::{
    instruction_input_row_bytes, instruction_input_sequence_auxiliary_storage_bytes,
    instruction_input_sequence_storage_bytes, instruction_input_weight_capacities,
    InstructionInputSequenceStorage, PendingInstructionInputPrimer,
    INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS, INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS,
    INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS,
};
pub use instruction_input::{
    InstructionInputRow, InstructionInputRows, InstructionInputSequence,
    InstructionInputSequenceConfig, InstructionInputStorageInitialization,
    INSTRUCTION_INPUT_COEFFICIENTS, INSTRUCTION_INPUT_TABLES,
};
pub(crate) use instruction_ra_sequence::InstructionRaSequenceStorage;
pub use instruction_ra_sequence::{
    InstructionRaMaterializeWidth, InstructionRaSequence, InstructionRaSequenceConfig,
    InstructionRaSequenceScratchLayout,
};
#[cfg(test)]
pub(crate) use instruction_read_raf::validate_bytecode_topology_admission;
#[doc(hidden)]
pub use instruction_read_raf::InstructionReadRafStage1Owner;
pub(crate) use instruction_read_raf::{
    instruction_read_raf_claim_and_count_rank, instruction_read_raf_stage1_claim_bytes,
    instruction_read_raf_stage1_device_bytes, instruction_read_raf_stage1_row_bytes,
    InstructionReadRafCompatibilityScatterConfig, InstructionReadRafCountOrder,
    InstructionReadRafDenseGroupedPlanes, InstructionReadRafDenseGroupedReceipt,
    InstructionReadRafFusedBytecodeReceipt, InstructionReadRafStage1ChunkWriter,
    InstructionReadRafStage1Lease, InstructionReadRafStage1Receipt,
    InstructionReadRafStage1Storage, PendingInstructionReadRafSourcePrimer,
    INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS, INSTRUCTION_READ_RAF_SEGMENTS,
};
#[cfg(feature = "allocative")]
pub(crate) use outer_remainder::OuterRegistersClaimCarrierSubmission;
#[cfg(feature = "test-utils")]
pub(crate) use outer_remainder::OuterRemainderStorageEvalStats;
pub(crate) use outer_remainder::{
    outer_remainder_sequence_max_buffer_bytes_with_config,
    outer_remainder_sequence_storage_bytes_with_config, OuterRegistersClaimCarrier,
    OuterRegistersClaimCarrierReceipt, OuterRemainderSequenceStorage,
    PendingOuterRegistersClaimCarrier,
};
pub use outer_remainder::{
    OuterRemainderPhase, OuterRemainderSequence, OuterRemainderSequenceConfig,
    OuterRemainderStorageInitialization, OuterRemainderStorageStats, OUTER_REMAINDER_OPENINGS,
};
pub use product5::{Product5Sequence, Product5SequenceConfig, PRODUCT5_FACTORS};
#[cfg(feature = "test-utils")]
pub use product_remainder::reference as product_remainder_reference;
pub(crate) use product_remainder::PendingProductRemainderInitialMessage;
pub use product_remainder::{
    ProductRemainderRow, ProductRemainderRowError, ProductRemainderRows, ProductRemainderSequence,
    ProductRemainderSequenceConfig, ProductRemainderShapeError, ProductRemainderSourceKind,
    ProductRemainderStorageLayout, PRODUCT_REMAINDER_MESSAGE_COLUMNS, PRODUCT_REMAINDER_OPENINGS,
    PRODUCT_REMAINDER_SIMD_WIDTH,
};
#[cfg(feature = "test-utils")]
pub use product_uniskip::reference as product_uniskip_reference;
pub use product_uniskip::{
    evaluate_product_uniskip_extensions_cpu, ProductUniskipExtendedNodes, ProductUniskipShapeError,
    PRODUCT_UNISKIP_EXTENDED_NODES, PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS,
    PRODUCT_UNISKIP_NODE_ORDER, PRODUCT_UNISKIP_SIMD_WIDTH,
};
pub(crate) use ram_hamming_sequence::RamHammingSequence;
pub(crate) use ram_ra_claim_reduction::RamRaClaimReductionSequence;
pub(crate) use ram_ra_sequence::RamRaSequence;
pub use ram_raf_evaluation::{
    dense_pushforward_oracle, split_equality as ram_raf_split_equality, split_pushforward_oracle,
    PendingRamRafSequence, RamRafAddress, RamRafAddressPlane, RamRafAffineTail, RamRafConfig,
    RamRafCounters, RamRafDeviceLimits, RamRafError, RamRafFoldParams, RamRafObservation,
    RamRafQuadraticMessage, RamRafSequence, RamRafShape, RamRafStoragePlan, RamRafTailOutput,
    ValidatedRamRafAddressPlane, RAM_RAF_ADDRESS_DOMAIN, RAM_RAF_AKITA_OFFSET,
    RAM_RAF_DEFAULT_TRACE_CUTOFF, RAM_RAF_FINALIZE_PIPELINE, RAM_RAF_FOLD_PIPELINE,
    RAM_RAF_INNER_LENGTH, RAM_RAF_INNER_LOG2, RAM_RAF_NO_ACCESS, RAM_RAF_SIMD_WIDTH,
    RAM_RAF_THREADS, RAM_RAF_TILE_ADDRESSES, RAM_RAF_TILE_COUNT,
};
#[cfg(feature = "test-utils")]
pub(crate) use ram_read_write::RamReadWritePreparationTiming;
pub(crate) use ram_read_write::{
    RamRafSegmentedAddressPlane, RamReadWriteDispatchTiming, RamReadWriteFinish,
    RamReadWriteSequence, SparseCycleProduct, RAM_READ_WRITE_CYCLE_TILE_LOG2,
};
pub(crate) use ram_val_sequence::RamValSequence;
#[cfg(feature = "test-utils")]
pub(crate) use registers_read_write::RegistersReadWriteCycleObservation;
pub(crate) use registers_read_write::{
    PendingRegistersReadWriteStage1Pipelines, RegistersReadWriteStage1ChunkWriter,
    RegistersReadWriteStage1Source, RegistersReadWriteStage1Storage,
};
pub(crate) use registers_val::PendingRegistersValFirstMessage;
pub use registers_val::{
    RegistersValDenseConfig, RegistersValFirstMessageConfig, RegistersValFirstMessageInvocation,
    RegistersValFirstTransitionInvocation, RegistersValSequence, RegistersValTransitionConfig,
};
pub(crate) use registers_val::{
    RegistersValInstructionSourceLease, RegistersValInstructionSourceRequest,
};
pub use spartan_outer_uniskip::{
    evaluate_spartan_outer_uniskip_cpu, SpartanOuterUniskipConfig, SpartanOuterUniskipInvocation,
    SpartanOuterUniskipRow, SpartanOuterUniskipRows, SPARTAN_OUTER_EXTENDED_NODES,
};
pub(crate) use spartan_outer_uniskip::{
    spartan_outer_uniskip_invocation_bytes, spartan_outer_uniskip_row_bytes,
    spartan_outer_uniskip_successor_row_bytes, OuterResidualArenaKey, OuterResidualReleaseReceipt,
    PendingSpartanStage1SourcePrimer, SpartanOuterUniskipColdRow, SpartanOuterUniskipSuccessorRow,
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

    copy_field_getters! { pub, { limbs: [u32; 4] }}

    pub const fn to_u128(self) -> u128 {
        (self.limbs[0] as u128)
            | ((self.limbs[1] as u128) << 32)
            | ((self.limbs[2] as u128) << 64)
            | ((self.limbs[3] as u128) << 96)
    }

    pub const fn is_canonical(self, offset: u32) -> bool {
        offset != 0 && self.to_u128() <= u128::MAX - offset as u128
    }

    pub fn from_jolt_field<F: CanonicalBytes>(value: &F) -> Self {
        debug_assert_eq!(F::NUM_BYTES, 16);
        let mut bytes = [0u8; 16];
        value.to_bytes_le(&mut bytes);
        Self::from_u128(u128::from_le_bytes(bytes))
    }

    pub fn into_jolt_field<F: CanonicalEncoding>(self) -> F {
        F::from_bytes_le_reduced(&self.to_u128().to_le_bytes())
    }
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
    #[error("the Metal pipeline cache is poisoned")]
    PipelineCachePoisoned,
    #[error("the Metal private-buffer pool is poisoned")]
    PrivateBufferPoolPoisoned,
    #[error("the Metal no-copy buffer cache is poisoned")]
    NoCopyBufferCachePoisoned,
    #[error("a non-noop dispatch requires at least one element")]
    EmptyInput,
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
    #[error("input lengths differ: lhs={lhs}, rhs={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("output[{index}] is not canonical for 2^128 - {offset}")]
    NonCanonicalOutput { index: usize, offset: u32 },
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
    #[error("invalid grouped InstructionReadRaf state: {0}")]
    InvalidInstructionReadRafGrouped(String),
    #[error("invalid co-produced RAM access collection: {0}")]
    InvalidRamAccessCollection(String),
    #[error("invalid bytecode read-RAF address configuration: {0}")]
    InvalidBytecodeReadRafAddressConfig(&'static str),
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
    #[error("RAM RA needs a power-of-two row count of at least 32, got {0}")]
    InvalidRamRaRows(usize),
    #[error("RAM RA factor-table storage has length {got}, expected {expected}")]
    RamRaStorageLength { expected: usize, got: usize },
    #[error("RAM RA split weights cover {covered} pairs, expected {expected}")]
    RamRaWeightShape { expected: usize, covered: usize },
    #[error(
        "RAM RA pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedRamRaExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "registers value evaluation needs a power-of-two cycle count of at least four, got {0}"
    )]
    InvalidRegistersValCycles(usize),
    #[error("registers value evaluation has {got} write indices, expected {expected}")]
    RegistersValIndexLength { expected: usize, got: usize },
    #[error(
        "registers value evaluation point shape mismatch: address_bits={address_bits}, cycle_bits={cycle_bits}, cycles={cycles}"
    )]
    RegistersValPointShape {
        address_bits: usize,
        cycle_bits: usize,
        cycles: usize,
    },
    #[error("registers value evaluation LT-low table has length {got}, expected {expected}")]
    RegistersValLtLength { expected: usize, got: usize },
    #[error("registers value evaluation dense state has {got} rows, expected {expected}")]
    RegistersValStateLength { expected: usize, got: usize },
    #[error(
        "registers value evaluation cannot continue split-LT binding from length {0}; hand off to the dense tail"
    )]
    RegistersValSplitLtExhausted(usize),
    #[error("invalid registers value-evaluation state: {0}")]
    InvalidRegistersValState(&'static str),
    #[error("registers value evaluation index {0} is outside the 128-register domain")]
    InvalidRegistersValIndex(u8),
    #[error(
        "registers value evaluation pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedRegistersValExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(transparent)]
    ProductRemainderShape(#[from] product_remainder::ProductRemainderShapeError),
    #[error("product remainder row {index} is invalid: {source}")]
    InvalidProductRemainderRow {
        index: usize,
        source: product_remainder::ProductRemainderRowError,
    },
    #[error("product remainder rows belong to Metal device {got}, expected {expected}")]
    ProductRemainderRowsDevice { expected: u64, got: u64 },
    #[error(transparent)]
    ProductUniskipShape(#[from] product_uniskip::ProductUniskipShapeError),
    #[error("product uni-skip rows belong to Metal device {got}, expected {expected}")]
    ProductUniskipRowsDevice { expected: u64, got: u64 },
    #[error(
        "product uni-skip pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedProductUniskipExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(transparent)]
    SpartanShiftPlan(#[from] spartan_shift::SpartanShiftPlanError),
    #[error(transparent)]
    SpartanShiftOracle(#[from] spartan_shift::SpartanShiftOracleError),
    #[error("invalid resident Spartan shift state: {0}")]
    InvalidSpartanShiftState(&'static str),
    #[error("Spartan shift row {row} could not be extracted: {message}")]
    SpartanShiftRowExtraction { row: usize, message: String },
    #[error("Spartan shift {name} buffer is on device {got}, expected {expected}")]
    SpartanShiftBufferDevice {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    #[error(
        "Spartan shift pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedSpartanShiftExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "Spartan shift fold needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    SpartanShiftThreadgroupMemory { requested: u64, maximum: u64 },
    #[error(transparent)]
    InstructionClaimShape(#[from] instruction_claim_reduction::InstructionClaimShapeError),
    #[error(transparent)]
    InstructionClaimOpening(#[from] instruction_claim_reduction::InstructionClaimOpeningError),
    #[error("invalid resident instruction claim-reduction state: {0}")]
    InvalidInstructionClaimState(&'static str),
    #[error(
        "instruction claim-reduction pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedInstructionClaimExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "instruction claim-reduction {phase} needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    InstructionClaimThreadgroupMemory {
        phase: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error(transparent)]
    RamRaf(#[from] ram_raf_evaluation::RamRafError),
    #[error("RAM RAF address plane belongs to Metal device {got}, expected {expected}")]
    RamRafRowsDevice { expected: u64, got: u64 },
    #[error(
        "RAM RAF shader reported {invalid_rows} invalid rows and {unsupported_dispatches} unsupported dispatches"
    )]
    RamRafDispatch {
        invalid_rows: u32,
        unsupported_dispatches: u32,
    },
    #[error("invalid resident RAM RAF state: {0}")]
    InvalidRamRafState(&'static str),
    #[error("invalid resident product remainder state: {0}")]
    InvalidProductRemainderState(&'static str),
    #[error("invalid resident RAM read-write state: {0}")]
    InvalidRamReadWriteState(&'static str),
    #[error("invalid registers read-write first-message state: {0}")]
    InvalidRegistersReadWriteState(&'static str),
    #[error("registers read-write pipeline has execution width {got}, expected {expected}")]
    UnsupportedRegistersReadWriteExecutionWidth { expected: usize, got: usize },
    #[error(
        "product remainder pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedProductRemainderExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid resident Instruction RA state: {0}")]
    InvalidInstructionRaState(&'static str),
    #[error("invalid resident RAM RA state: {0}")]
    InvalidRamRaState(&'static str),
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
    #[error("Booleanity address selector tiles must contain 1..=6 selectors, got {0}")]
    InvalidBooleanityAddressSelectorTile(usize),
    #[error("Booleanity address inner split must be in 0..=16, got {0}")]
    InvalidBooleanityAddressInnerLog2(usize),
    #[error("Booleanity address finalization needs 256, 512, 768, or 1024 threads, got {0}")]
    InvalidBooleanityAddressFinalizeWidth(usize),
    #[error(
        "Booleanity address needs {requested} bytes of threadgroup memory, device limit is {maximum}"
    )]
    BooleanityAddressThreadgroupMemory { requested: u64, maximum: u64 },
    #[error(
        "booleanity pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedBooleanityExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("outer remainder needs a power-of-two cycle count of at least four, got {0}")]
    InvalidOuterRemainderRows(usize),
    #[error(
        "outer remainder explicit prefix has {explicit} rows, exceeding logical length {logical}"
    )]
    OuterRemainderExplicitRows { explicit: usize, logical: usize },
    #[error("outer remainder rows belong to Metal device {got}, but the kernel uses {expected}")]
    OuterRemainderRowDevice { expected: u64, got: u64 },
    #[error("invalid outer remainder configuration: {0}")]
    InvalidOuterRemainderConfig(&'static str),
    #[error("invalid outer remainder state: expected {expected}, got {got}")]
    InvalidOuterRemainderState {
        expected: &'static str,
        got: &'static str,
    },
    #[error(
        "outer remainder {phase} weights have e_in={e_in}, e_out={e_out}; expected product {expected}"
    )]
    OuterRemainderWeightShape {
        phase: &'static str,
        expected: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error("outer remainder {name} storage has capacity {capacity}, got {got} values")]
    OuterRemainderStorageLength {
        name: &'static str,
        capacity: usize,
        got: usize,
    },
    #[error("outer remainder CPU tail needs {expected} Az/Bz values, got az={az}, bz={bz}")]
    OuterRemainderTailLength {
        expected: usize,
        az: usize,
        bz: usize,
    },
    #[error(
        "outer remainder pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedOuterRemainderExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "outer remainder opening kernel needs {requested} bytes of threadgroup memory, device limit is {maximum}"
    )]
    OuterRemainderThreadgroupMemory { requested: u64, maximum: u64 },
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

impl MetalError {
    pub(crate) fn is_capacity_error(&self) -> bool {
        matches!(
            self,
            Self::InputTooLong(_)
                | Self::BufferTooLong { .. }
                | Self::WorkingSetTooLarge { .. }
                | Self::InstructionClaimShape(
                    instruction_claim_reduction::InstructionClaimShapeError::BufferLengthLimit { .. }
                )
        )
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use super::{validate_working_set, Fp128, MetalError, OFFSET_275};

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
    fn capacity_errors_are_safe_pre_submit_fallbacks() {
        for error in [
            MetalError::InputTooLong(usize::MAX),
            MetalError::BufferTooLong {
                requested: 101,
                maximum: 100,
            },
            MetalError::WorkingSetTooLarge {
                current: 60,
                additional: 41,
                maximum: 100,
            },
        ] {
            assert!(error.is_capacity_error());
        }
        assert!(!MetalError::EmptyInput.is_capacity_error());
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
}
