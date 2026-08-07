//! First-principles successor packet for the RAM value-check Metal kernel.
//!
//! The sparse first-message path runs as a protocol-inert parity shadow. The
//! optimized CPU kernel remains authoritative while production timing decides
//! whether later rounds merit a device implementation.

pub mod abi;
pub mod model;
pub mod oracle;
mod runtime;

pub const SOURCE: &str = include_str!("shader.metal");

pub use abi::{
    IncrementAccessRow, IncrementAccessSource, RamValActivePair, RamValBufferRange,
    RamValFirstMessageBufferLengths, RamValFirstMessageParams, RamValLaunch,
    RamValReductionBuffers, RamValReductionParams, RamValSparseFirstMessageParams,
    RamValSuccessorDispatchError, RamValSuccessorRowError,
};
pub use model::{
    admission_decision, sparse_screen_class, speed_screen_decision, target_work_plan,
    ActivityProvenance, ActivityProvenanceRejection, AdmissionDecision, CandidateEvidence,
    CompiledCaptureEvidence, CompiledCaptureRejection, CompiledPhaseResources, PhaseLatencySamples,
    PhaseRoofRejection, ProducerEvidence, RoofBounds, SparseScreenClass, SuccessorPhase,
};
pub use runtime::{
    PendingRamValSparseFirstMessage, RamValSparseFirstMessage, RamValSparseFirstMessageStats,
};

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests;
