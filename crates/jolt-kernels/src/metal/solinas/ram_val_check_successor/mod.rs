//! First-principles successor packet for the RAM value-check Metal kernel.
//!
//! This module is intentionally not registered yet. It freezes the relation,
//! producer lease, performance model, independent oracle, and the first sparse
//! message shader before proof-stage integration changes any shared code.

pub mod abi;
pub mod model;
pub mod oracle;

pub use abi::{
    IncrementAccessRow, IncrementAccessSource, RamValBufferRange, RamValFirstMessageBufferLengths,
    RamValFirstMessageParams, RamValLaunch, RamValReductionBuffers, RamValReductionParams,
    RamValSuccessorDispatchError, RamValSuccessorRowError,
};
pub use model::{
    admission_decision, sparse_screen_class, speed_screen_decision, target_work_plan,
    ActivityProvenance, ActivityProvenanceRejection, AdmissionDecision, CandidateEvidence,
    CompiledCaptureEvidence, CompiledCaptureRejection, CompiledPhaseResources, PhaseLatencySamples,
    PhaseRoofRejection, ProducerEvidence, RoofBounds, SparseScreenClass, SuccessorPhase,
};

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests;
