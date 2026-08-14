mod api;
mod artifact;
mod plan;
mod registers_claim;
#[cfg(feature = "test-utils")]
mod sealed;
mod sequence;
mod shader;
mod storage;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;

pub use api::{
    OuterRemainderDispatchCounts, OuterRemainderPhase, OuterRemainderSequenceConfig,
    OuterRemainderStorageInitialization, OuterRemainderStorageInitializationStats,
    OuterRemainderStorageStats, OUTER_REMAINDER_OPENINGS,
};
pub use artifact::OuterBindingPlan;
#[cfg(feature = "test-utils")]
pub use artifact::OuterKernelArtifact;
pub(crate) use plan::{
    outer_remainder_sequence_max_buffer_bytes_with_config,
    outer_remainder_sequence_storage_bytes_with_config,
};
pub(crate) use registers_claim::{OuterRegistersClaimCarrier, OuterRegistersClaimCarrierReceipt};
#[cfg(feature = "test-utils")]
pub use sealed::SealedOuterArtifact;
pub use sequence::OuterRemainderSequence;
pub(crate) use sequence::{
    OuterRegistersClaimCarrierJoinStats, OuterRegistersClaimCarrierSubmission,
    PendingOuterRegistersClaimCarrier,
};
pub(super) use shader::{PADDED_56_SOURCE, SOURCE};
pub(crate) use storage::OuterRemainderSequenceStorage;
