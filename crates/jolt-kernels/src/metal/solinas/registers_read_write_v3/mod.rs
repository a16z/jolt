//! Checked host foundation for the registers read/write v3 Metal sequence.
//!
//! The implementation map is:
//!
//! - [`abi`] owns CSR geometry, plane layouts, and device allocation receipts.
//! - [`bcsr`] defines the padded CSR-256 producer ABI and its scalar adapter.
//! - [`execution_abi`] fixes shader bindings, parameters, and arena ownership.
//! - [`execution_model`] reconstructs the BCSR-256 log-26 execution census.
//! - [`owner`] constructs and certifies CSR-256 register state flow.
//! - [`oracle`] contains independent dense and CSR-native relation evaluators.
//! - [`model`] retains the pre-BCSR comparison baseline.
//!
//! The ordered-prefix digest is opaque here. The stage-1 producer computes it;
//! this package only binds later receipts to the exact 256-bit value. Command
//! completion is likewise represented as admitted allocation metadata until
//! the Metal runtime maps it to a completed command serial.

#![expect(
    dead_code,
    unused_imports,
    reason = "the v3 foundation is hidden until the runtime slice is wired"
)]

mod abi;
mod bcsr;
mod execution_abi;
mod execution_model;
mod model;
mod oracle;
mod owner;

pub(crate) use abi::{
    OrderedPrefixDigest, PlaneAllocation, PlaneShape, RegisterCsrCensus, RegisterEventCounts,
    RegisterGeometry, RegisterPlaneAllocations, RegisterPlaneLayout, RegisterProducerIdentity,
    REGISTER_ADDRESS_BITS, REGISTER_CSR_BLOCK_CYCLES, REGISTER_CSR_COLUMNS, REGISTER_FP128_BYTES,
    REGISTER_LOG26_CENSUS, REGISTER_LOG26_CSR_BYTES, REGISTER_LOG26_PRODUCER_BYTES,
};
pub(crate) use bcsr::{
    RegisterBcsr256, RegisterBcsr256Parts, RegisterBcsrGeometry, RegisterBcsrLayout,
    RegisterBcsrPlaneProvenance, RegisterBcsrPlaneProvenances, RegisterBcsrPlaneShape,
    RegisterBcsrReadEvent, RegisterBcsrReceipt, RegisterBcsrSourceProvenance,
    RegisterBcsrStateFlowCertificate, RegisterBcsrWriteEvent, RegistersValInputReceipt,
    REGISTER_ABSENT_INDEX, REGISTER_BCSR_OFFSET_ENTRIES, REGISTER_BCSR_POSITION_SLOTS,
};
pub(crate) use execution_abi::{
    Arena, ArenaLifetime, BufferAccess, BufferBinding, DenseRoundParams, DenseState,
    DispatchGeometry, HistogramParams, HostSchedule, LifetimeDisposition, PhaseDescriptor,
    PipelineDescriptor, PipelineReadiness, RawCoefficientParams, RawReplayParams,
    RawRoundZeroParams, ReductionParams, RegistersValHandoff, SequencePoint, ARENA_LIFETIMES,
    DENSE_DESCRIPTOR, HISTOGRAM_DESCRIPTOR, PHASES, PIPELINES, RAW_COEFFICIENT_DESCRIPTOR,
    RAW_REPLAY_DESCRIPTOR, RAW_ROUND_ZERO_DESCRIPTOR, REDUCTION_DESCRIPTOR, REGISTERS_VAL_HANDOFF,
    SOURCE as EXECUTION_SOURCE,
};
pub(crate) use execution_model::{
    trace_peak_logical_bytes, ExecutionWork, LaunchAccounting, Log26ExecutionModel, ProductCensus,
    RawRoundProductCensus, TimeBudget, TraceExecutionPlan, ANALYTICAL_EXECUTION_HIGH_NS,
    ANALYTICAL_EXECUTION_LOW_NS, CPU_BASELINE_NS, CURRENT_CPU_FALLBACK_NS,
    LATEST_DIAGNOSTIC_CPU_NS, LOG26_PEAK_LOGICAL_BYTES, LOG26_RAW_ROUND_PRODUCTS,
    PRODUCER_PURSUIT_CAP_NS, TIME_BUDGETS,
};
pub(crate) use model::{
    GateReport, LifecycleProjection, Log26Accounting, PhaseWork, RoofRates, SpeedupGate,
    M4_MAX_ROOF_RATES,
};
pub(crate) use oracle::{
    CycleRoundReference, DenseCell, DenseRegisterRelation, Round8Junction, RoundZeroBasisSums,
    RoundZeroInfinityEvent, RoundZeroPairEvents, RoundZeroTernaryBasis, RoundZeroValueEvent,
    SignedU64, SparseRegisterRelation, ROUND_ZERO_CONSTANT_BASIS, ROUND_ZERO_GAMMA_BASIS,
    ROUND_ZERO_GAMMA_SQ_BASIS,
};
pub(crate) use owner::{
    CertifiedRegisterOwner, CertifiedRegisterOwnerReceipt, RegisterCsr256, RegisterCsr256Parts,
    RegisterRead, RegisterRow, RegisterStateFlowCertificate, RegisterWrite,
};

use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum RegistersRwV3Error {
    #[error("register cycle count {0} must be a nonzero power of two in the u32 domain")]
    InvalidCycleCount(usize),
    #[error("register BCSR cycle count {0} must be nonzero and fit in u32")]
    InvalidBcsrCycleCount(usize),
    #[error("register v3 size arithmetic overflowed while computing {0}")]
    SizeOverflow(&'static str),
    #[error("register {plane} census has {count} events for {cycles} cycles")]
    InvalidEventCount {
        plane: &'static str,
        cycles: usize,
        count: usize,
    },
    #[error("register v3 {0} identity must be nonzero")]
    MissingIdentity(&'static str),
    #[error("register v3 ordered-prefix digest must be nonzero")]
    ZeroOrderedPrefixDigest,
    #[error("register v3 producer has {got} cycles, expected {expected}")]
    ProducerCycleMismatch { expected: usize, got: usize },
    #[error("register v3 {plane} allocation is on device {got}, expected {expected}")]
    PlaneDeviceMismatch {
        plane: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("register v3 {plane} generation is {got}, expected {expected}")]
    PlaneGenerationMismatch {
        plane: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("register v3 {plane} allocation has not completed initialization")]
    PlaneInitializationIncomplete { plane: &'static str },
    #[error(
        "register v3 {plane} allocation is {got_elements} elements/{got_bytes} bytes, expected {expected_elements} elements/{expected_bytes} bytes"
    )]
    PlaneShape {
        plane: &'static str,
        expected_elements: usize,
        got_elements: usize,
        expected_bytes: usize,
        got_bytes: usize,
    },
    #[error("register v3 allocation identity {identity} is reused")]
    DuplicateAllocationIdentity { identity: usize },
    #[error("register v3 receipt device is {got}, expected {expected}")]
    ReceiptDeviceMismatch { expected: u64, got: u64 },
    #[error("register v3 receipt generation is {got}, expected {expected}")]
    ReceiptGenerationMismatch { expected: u64, got: u64 },
    #[error("register v3 receipt ordered-prefix digest changed")]
    ReceiptDigestMismatch,
    #[error("register v3 {plane} length is {got}, expected {expected}")]
    PlaneLength {
        plane: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("register v3 {plane} offsets start at {got}, expected zero")]
    OffsetStart { plane: &'static str, got: u32 },
    #[error("register v3 {plane} offsets decrease at header {header}: {start} to {end}")]
    OffsetOrder {
        plane: &'static str,
        header: usize,
        start: u32,
        end: u32,
    },
    #[error("register v3 {plane} terminal offset is {got}, expected {expected}")]
    OffsetTerminal {
        plane: &'static str,
        expected: usize,
        got: u32,
    },
    #[error("register v3 {plane} block {block} terminal offset is {got}, maximum {maximum}")]
    BcsrOffsetTerminal {
        plane: &'static str,
        block: usize,
        maximum: usize,
        got: u16,
    },
    #[error("register v3 {plane} block {block} has nonzero padding at slot {slot}")]
    BcsrNonzeroPadding {
        plane: &'static str,
        block: usize,
        slot: usize,
    },
    #[error("register v3 {plane} positions are not increasing at header {header}")]
    PositionOrder { plane: &'static str, header: usize },
    #[error("register v3 {plane} position {position} exceeds block {block} length {block_len}")]
    PositionOutOfBlock {
        plane: &'static str,
        block: usize,
        block_len: usize,
        position: u8,
    },
    #[error("register v3 {plane} has more than one event at cycle {cycle}")]
    DuplicateCycleEvent { plane: &'static str, cycle: usize },
    #[error(
        "register v3 rd index at cycle {cycle} is {got}, expected {expected} from the write plane"
    )]
    RdIndexMismatch { cycle: usize, expected: u8, got: u8 },
    #[error("register v3 block {block} register {register} starts at {got}, expected {expected}")]
    BlockStateMismatch {
        block: usize,
        register: usize,
        expected: u64,
        got: u64,
    },
    #[error("register v3 {access} index {register} at cycle {cycle} is out of range")]
    InvalidRegister {
        cycle: usize,
        access: &'static str,
        register: u8,
    },
    #[error(
        "register v3 {access} value at cycle {cycle}, register {register}, is {got}, expected {expected}"
    )]
    ReadValueMismatch {
        cycle: usize,
        access: &'static str,
        register: u8,
        expected: u64,
        got: u64,
    },
    #[error(
        "register v3 rd pre-value at cycle {cycle}, register {register}, is {got}, expected {expected}"
    )]
    WritePreValueMismatch {
        cycle: usize,
        register: u8,
        expected: u64,
        got: u64,
    },
    #[error("register v3 {plane} event count exceeds u32")]
    EventCountOverflow { plane: &'static str },
    #[error("register v3 {name} index {index} is out of range {length}")]
    IndexOutOfRange {
        name: &'static str,
        index: usize,
        length: usize,
    },
    #[error("register v3 {name} has length {got}, expected {expected}")]
    InputLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("register v3 rd increment disagrees with CSR state flow at cycle {cycle}")]
    IncrementMismatch { cycle: usize },
    #[error("register v3 cycle round is unavailable with {remaining_rows} rows remaining")]
    CycleRoundUnavailable { remaining_rows: usize },
    #[error("register v3 round-8 junction requested after {rounds_bound} binds")]
    JunctionRoundMismatch { rounds_bound: usize },
    #[error("register v3 log-26 analytical constants failed their checked reconstruction")]
    AnalyticalCensusMismatch,
    #[error("register v3 roof-model parameter {0} must be finite and positive")]
    InvalidRoofParameter(&'static str),
    #[error("registers-value resident handoff rejects {0} rows")]
    InvalidRegistersValHandoff(usize),
    #[error("registers RW v3 execution round {0} is outside its phase")]
    InvalidExecutionRound(u32),
    #[error("registers RW v3 execution supports target trace logs 26 through 28, got {0}")]
    InvalidExecutionLogT(u32),
    #[error("registers RW v3 execution parameter {0} is invalid")]
    InvalidExecutionParameter(&'static str),
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "fixed unit-test fixtures use direct assertions and unwraps"
)]
mod tests;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "fixed BCSR fixtures use direct assertions and unwraps"
)]
mod bcsr_tests;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "fixed execution-model fixtures use direct assertions and unwraps"
)]
mod execution_tests;
