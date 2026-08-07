//! Canonical 128-bit Solinas-field arithmetic on Metal.
//!
//! [`Fp128`] is the buffer ABI, not a host field implementation. Arithmetic is
//! performed by the shader specialized for `2^128 - C`; host callers supply
//! canonical values for the selected offset.

use std::{cell::Cell, slice, time::Duration};

use jolt_field::FixedBytes;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

mod address_raf;
mod address_raf_direct;
mod address_sequence;
mod address_suffix;
mod address_suffix_full;
mod booleanity;
mod booleanity_address;
pub mod booleanity_address_successor;
#[doc(hidden)]
pub mod booleanity_address_v2;
mod bytecode_cycle;
#[doc(hidden)]
pub mod bytecode_read_raf;
#[doc(hidden)]
pub mod bytecode_read_raf_v2;
#[doc(hidden)]
pub mod bytecode_read_raf_v3;
mod bytecode_row;
pub mod half_width_probe;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub mod hamming_weight_claim_reduction;
#[cfg(not(feature = "test-utils"))]
mod hamming_weight_claim_reduction;
pub mod hamming_weight_claim_reduction_successor;
#[doc(hidden)]
pub mod hamming_weight_claim_reduction_v2;
pub mod instruction_claim_reduction;
mod instruction_input;
pub mod instruction_input_successor;
mod instruction_ra_sequence;
mod instruction_ra_virtualization;
#[doc(hidden)]
pub mod instruction_read_raf_producer;
#[doc(hidden)]
pub mod instruction_read_raf_v2;
#[doc(hidden)]
pub mod instruction_read_raf_v3;
mod outer_remainder;
mod product5;
mod product_remainder;
mod product_uniskip;
#[doc(hidden)]
pub mod ram_cycle_family_v3;
#[doc(hidden)]
pub mod ram_family_v2;
mod ram_output_check;
pub mod ram_ra_claim_reduction;
mod ram_raf_evaluation;
mod ram_val_check;
pub mod ram_val_check_successor;
mod registers;
pub mod registers_claim_reduction;
mod registers_read_write;
mod registers_read_write_dense;
mod registers_read_write_v3;
mod registers_val;
#[doc(hidden)]
pub mod registers_val_claim_v2;
pub mod registers_val_evaluation_backend;
mod runtime;
mod source;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub mod spartan_outer_successor;
mod spartan_outer_uniskip;
pub mod spartan_shift;

pub(crate) use runtime::validate_working_set;
#[cfg(feature = "test-utils")]
pub use runtime::SolinasMetalCompilationStats;
use runtime::{buffer_from_slice, command_buffer_timestamp};
pub use runtime::{DeviceInfo, PipelineLimits, SolinasMetal};

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
pub use booleanity::{
    BooleanityRow, BooleanitySelector, BooleanitySequence, BooleanitySequenceConfig,
};
pub use booleanity::{BooleanityRows, HammingHotRows};
pub use booleanity_address::{BooleanityAddressPushforward, BooleanityAddressPushforwardConfig};
pub use booleanity_address_successor::{
    BooleanityAddressSuccessorConfig, BooleanityAddressSuccessorInvocation,
    BooleanityAddressSuccessorRuntimeError,
};
pub use bytecode_cycle::{
    BytecodeCycleSequence, BytecodeCycleSequenceConfig, BytecodeCycleTables,
    BytecodeCycleTablesMut, BYTECODE_CYCLE_SAMPLES, BYTECODE_CYCLE_TABLES,
};
pub(crate) use bytecode_row::{BytecodeCycleRowInputs, BytecodeCycleRowSequence};
pub use half_width_probe::{
    HalfWidthDomain, HalfWidthOperand, HalfWidthProbe, HalfWidthProbeInvocation,
    HalfWidthProbeShape, HALF_WIDTH_AKITA_OFFSET, MINIMUM_HALF_WIDTH_PRODUCTS_PER_SECOND,
    TARGET_CHAIN_ELEMENTS, TARGET_CHAIN_ITERATIONS,
};
pub use hamming_weight_claim_reduction::HammingWeightSuccessorError;
pub use hamming_weight_claim_reduction_successor::{
    HammingWeightRetainedConfig, HammingWeightRetainedInvocation, HammingWeightRetainedRuntimeError,
};
pub(crate) use instruction_input::{
    instruction_input_row_bytes, instruction_input_sequence_storage_bytes,
    instruction_input_weight_capacities, InstructionInputSequenceStorage,
    PendingInstructionInputPrimer, INSTRUCTION_INPUT_PRIMER_E_IN_ELEMENTS,
    INSTRUCTION_INPUT_PRIMER_E_OUT_ELEMENTS, INSTRUCTION_INPUT_PRIMER_SOURCE_ELEMENTS,
};
pub use instruction_input::{
    InstructionInputFirstTransition, InstructionInputPrimerStats, InstructionInputRow,
    InstructionInputRows, InstructionInputSequence, InstructionInputSequenceConfig,
    InstructionInputStorageInitialization, InstructionInputStorageInitializationStats,
    InstructionInputSuccessorDenseMessageStats, InstructionInputSuccessorMaterializeStats,
    InstructionInputSuccessorTransitionStats, INSTRUCTION_INPUT_COEFFICIENTS,
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
pub(crate) use outer_remainder::{
    outer_remainder_sequence_max_buffer_bytes_with_config,
    outer_remainder_sequence_storage_bytes_with_config, OuterRemainderSequenceStorage,
};
pub use outer_remainder::{
    OuterBindingPlan, OuterRemainderDispatchCounts, OuterRemainderPhase, OuterRemainderSequence,
    OuterRemainderSequenceConfig, OuterRemainderStorageInitialization,
    OuterRemainderStorageInitializationStats, OuterRemainderStorageStats, OUTER_REMAINDER_OPENINGS,
};
#[cfg(feature = "test-utils")]
pub use outer_remainder::{OuterKernelArtifact, SealedOuterArtifact};
pub use product5::{
    DenseTransitionError, DenseTransitionInvocation, DenseTransitionObservation,
    DenseTransitionParams, DenseTransitionTile, Product5Config, Product5Invocation,
    Product5Sequence, Product5SequenceConfig, PRODUCT5_FACTORS,
};
#[cfg(feature = "test-utils")]
pub use product_remainder::reference as product_remainder_reference;
pub(crate) use product_remainder::PendingProductRemainderInitialMessage;
pub use product_remainder::{
    ProductRemainderRow, ProductRemainderRowError, ProductRemainderRows, ProductRemainderSequence,
    ProductRemainderSequenceConfig, ProductRemainderShapeError, ProductRemainderStorageLayout,
    PRODUCT_REMAINDER_MESSAGE_COLUMNS, PRODUCT_REMAINDER_OPENINGS, PRODUCT_REMAINDER_SIMD_WIDTH,
};
#[cfg(feature = "test-utils")]
pub use product_uniskip::reference as product_uniskip_reference;
pub use product_uniskip::{
    evaluate_product_uniskip_extensions_cpu, ProductUniskipConfig, ProductUniskipExtendedNodes,
    ProductUniskipInvocation, ProductUniskipKnownNodes, ProductUniskipShapeError,
    ProductUniskipStorageLayout, PRODUCT_UNISKIP_EXTENDED_NODES,
    PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS, PRODUCT_UNISKIP_NODE_ORDER, PRODUCT_UNISKIP_SIMD_WIDTH,
};
pub use ram_output_check::{
    fold_low_prefix as ram_output_check_fold_low_prefix,
    fold_public_segments as ram_output_check_fold_public_segments,
    fold_u64_low_prefix as ram_output_check_fold_u64_low_prefix,
    folded_range_mask as ram_output_check_folded_range_mask,
    low_binding_weights as ram_output_check_low_binding_weights, DenseRamOutputOracle,
    RamOutputCheckCost, RamOutputCheckFold, RamOutputCheckFoldParams, RamOutputCheckHybridPlan,
    RamOutputCheckPlanError, RamOutputCheckStorage, RamOutputPublicSegment,
    ResidentRamFinalMetadata, ResidentRamFinalValues, RAM_OUTPUT_CHECK_COMPONENT_GATE_NS,
    RAM_OUTPUT_CHECK_FIVE_X_CAP_NS, RAM_OUTPUT_CHECK_FOLD_PIPELINE,
    RAM_OUTPUT_CHECK_REDUCE_PIPELINE, RAM_OUTPUT_CHECK_RELATION_CPU_NS,
    RAM_OUTPUT_CHECK_RELATION_FIVE_X_CAP_NS, RAM_OUTPUT_CHECK_SIMD_WIDTH,
    RAM_OUTPUT_CHECK_TARGET_ADDRESSES, RAM_OUTPUT_CHECK_TARGET_CPU_NS,
    RAM_OUTPUT_CHECK_TARGET_LOG_K, RAM_OUTPUT_CHECK_TARGET_MASK_END,
    RAM_OUTPUT_CHECK_TARGET_MASK_START,
};
pub use ram_raf_evaluation::{
    address_opening_point as ram_raf_address_opening_point, dense_pushforward_oracle,
    split_equality as ram_raf_split_equality, split_pushforward_oracle, tiled_pushforward_oracle,
    PendingRamRafSequence, RamRafAddress, RamRafAddressPlane, RamRafAffineTail, RamRafConfig,
    RamRafCostModel, RamRafCounters, RamRafDecision, RamRafDeviceLimits, RamRafError,
    RamRafEvidence, RamRafExecution, RamRafFoldParams, RamRafMeasuredResult, RamRafObservation,
    RamRafProjection, RamRafQuadraticMessage, RamRafSequence, RamRafShape, RamRafStoragePlan,
    RamRafSubmissionStats, RamRafTailOutput, RamRafTopology, ValidatedRamRafAddressPlane,
    RAM_RAF_ADDRESS_DOMAIN, RAM_RAF_AKITA_OFFSET, RAM_RAF_DEFAULT_TRACE_CUTOFF,
    RAM_RAF_FINALIZE_PIPELINE, RAM_RAF_FIXED_PROJECTION_NS, RAM_RAF_FOLD_PIPELINE,
    RAM_RAF_FOLD_REDESIGN_NS, RAM_RAF_INNER_LENGTH, RAM_RAF_INNER_LOG2, RAM_RAF_NO_ACCESS,
    RAM_RAF_PURSUIT_NS, RAM_RAF_SIMD_WIDTH, RAM_RAF_TARGET_CPU_NS, RAM_RAF_TARGET_CPU_PREPARE_NS,
    RAM_RAF_TARGET_CPU_TAIL_NS, RAM_RAF_TARGET_FIVE_X_NS, RAM_RAF_TARGET_LOG_T,
    RAM_RAF_TARGET_ROWS, RAM_RAF_THREADS, RAM_RAF_TILE_ADDRESSES, RAM_RAF_TILE_COUNT,
};
#[cfg(feature = "test-utils")]
pub use ram_val_check::oracle as ram_val_check_oracle;
pub use ram_val_check::{
    RamValCheckConfig, RamValCheckDenseRow, RamValCheckNativeRow, RamValCheckPlan,
    RamValCheckRowError, RamValCheckRows, RamValCheckSequence, RamValCheckShapeError,
    RamValCheckStorageLayout, RAM_VAL_CHECK_DEFAULT_CPU_TAIL_ELEMENTS,
    RAM_VAL_CHECK_FIVE_X_GATE_NS, RAM_VAL_CHECK_MESSAGE_COLUMNS, RAM_VAL_CHECK_NO_ACCESS,
    RAM_VAL_CHECK_SIMD_WIDTH, RAM_VAL_CHECK_TARGET_CPU_NS,
};
pub use ram_val_check_successor::{
    PendingRamValSparseFirstMessage, RamValActivePair, RamValSparseFirstMessage,
    RamValSparseFirstMessageStats,
};
pub use registers::{
    CertifiedRegisterOwner, RdIncrement, RdIncrementActivity, RegisterCsr256, RegisterCsr256Parts,
    RegisterCsrCensus, RegisterEventCounts, RegisterOwnerError, RegisterOwnerRead,
    RegisterOwnerRow, RegisterOwnerWrite, RegisterStateFlowCertificate, REGISTER_CSR_BLOCK_CYCLES,
    REGISTER_CSR_COLUMNS, REGISTER_CSR_NON_AUTHORITATIVE_LOG_T_26_CENSUS,
};
pub use registers_read_write::{
    RegisterAccessRow, RegistersReadWriteFirstMessageInvocation, RegistersReadWriteMessageConfig,
    RegistersReadWriteSecondMessageInvocation,
};
pub use registers_read_write_dense::{
    RegistersRwDenseRoundInvocation, RegistersRwDenseRoundStorage, RegistersRwDenseStateWords,
    REGISTERS_RW_DENSE_COLUMNS,
};
pub(crate) use registers_val::{PendingRegistersValFirstMessage, RegistersValFirstMessageStats};
pub use registers_val::{
    RegistersValDenseConfig, RegistersValFirstMessageConfig, RegistersValFirstMessageInvocation,
    RegistersValFirstTransitionInvocation, RegistersValSequence, RegistersValTransitionConfig,
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

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("no Metal device is available")]
    DeviceUnavailable,
    #[error("Solinas offset must be nonzero")]
    InvalidOffset,
    #[error("invalid OuterRemainder runtime artifact source")]
    InvalidOuterArtifactSource,
    #[error("invalid sealed OuterRemainder artifact: {0}")]
    InvalidSealedOuterArtifact(String),
    #[error("OuterRemainder artifact and runtime binding plans differ")]
    OuterArtifactBindingPlanMismatch,
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
    #[error(transparent)]
    HalfWidthProbe(#[from] half_width_probe::HalfWidthProbeError),
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
    #[error("invalid grouped InstructionReadRaf state: {0}")]
    InvalidInstructionReadRafGrouped(String),
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
    #[error("registers read/write needs a power-of-two row count of at least two, got {0}")]
    InvalidRegistersReadWriteRows(usize),
    #[error("registers read/write inc has length {got}, expected {expected}")]
    RegistersReadWriteIncLength { expected: usize, got: usize },
    #[error("registers read/write split weights cover {covered} pairs, expected {expected}")]
    RegistersReadWriteWeightShape { expected: usize, covered: usize },
    #[error("registers read/write index {0} is outside the 128-register domain")]
    InvalidRegistersReadWriteIndex(u8),
    #[error(
        "registers read/write pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedRegistersReadWriteExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(transparent)]
    RegistersReadWriteDenseAbi(#[from] registers_read_write_dense::RegistersRwDenseAbiError),
    #[error(
        "registers read/write needs {requested} bytes of threadgroup memory, device limit is {maximum}"
    )]
    RegistersReadWriteThreadgroupMemory { requested: u64, maximum: u64 },
    #[error("invalid resident registers read/write state: {0}")]
    InvalidRegistersReadWriteState(&'static str),
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
    RamOutputCheck(#[from] ram_output_check::RamOutputCheckPlanError),
    #[error("RAM output-check values belong to Metal device {got}, expected {expected}")]
    RamOutputCheckValuesDevice { expected: u64, got: u64 },
    #[error("invalid resident RAM output-check state: {0}")]
    InvalidRamOutputCheckState(&'static str),
    #[error(
        "RAM output-check pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedRamOutputCheckExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "RAM output-check fold needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    RamOutputCheckThreadgroupMemory { requested: u64, maximum: u64 },
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
    #[error(transparent)]
    HammingWeightSuccessor(#[from] hamming_weight_claim_reduction::HammingWeightSuccessorError),
    #[error(
        "Hamming-weight pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedHammingWeightExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "Hamming-weight pipeline `{pipeline}` needs {requested} threads, pipeline maximum is {maximum}"
    )]
    HammingWeightThreadgroupLimit {
        pipeline: &'static str,
        requested: usize,
        maximum: usize,
    },
    #[error(
        "Hamming-weight pipeline `{pipeline}` needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    HammingWeightThreadgroupMemory {
        pipeline: &'static str,
        requested: u64,
        maximum: u64,
    },
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
    InstructionInputSuccessor(#[from] instruction_input_successor::InstructionInputSuccessorError),
    #[error(
        "InstructionInput successor dense message needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    InstructionInputSuccessorThreadgroupMemory { requested: u64, maximum: u64 },
    #[error(transparent)]
    RamValCheckShape(#[from] ram_val_check::RamValCheckShapeError),
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
    #[error("RAM value-check rows belong to Metal device {got}, expected {expected}")]
    RamValCheckRowsDevice { expected: u64, got: u64 },
    #[error(
        "RAM value-check pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedRamValCheckExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "RAM value-check {phase} needs {requested} bytes of threadgroup memory, device maximum is {maximum}"
    )]
    RamValCheckThreadgroupMemory {
        phase: &'static str,
        requested: u64,
        maximum: u64,
    },
    #[error("invalid resident RAM value-check state: {0}")]
    InvalidRamValCheckState(&'static str),
    #[error("invalid resident product remainder state: {0}")]
    InvalidProductRemainderState(&'static str),
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

impl SolinasMetal {
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
