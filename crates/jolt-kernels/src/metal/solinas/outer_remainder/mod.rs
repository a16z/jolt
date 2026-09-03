mod api;
mod plan;
mod registers_claim;
mod sequence;
mod shader;
mod storage;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;

pub use api::{
    OuterRemainderPhase, OuterRemainderSequenceConfig, OuterRemainderStorageInitialization,
    OuterRemainderStorageStats, OUTER_REMAINDER_OPENINGS,
};
pub(crate) use plan::{
    outer_remainder_sequence_max_buffer_bytes_with_config,
    outer_remainder_sequence_storage_bytes_with_config,
};
pub(crate) use registers_claim::{OuterRegistersClaimCarrier, OuterRegistersClaimCarrierReceipt};
#[cfg(feature = "allocative")]
pub(crate) use sequence::OuterRegistersClaimCarrierSubmission;
pub use sequence::OuterRemainderSequence;
pub(crate) use sequence::PendingOuterRegistersClaimCarrier;
pub(super) use shader::SOURCE;
pub(crate) use storage::OuterRemainderSequenceStorage;
#[cfg(feature = "test-utils")]
pub(crate) use storage::OuterRemainderStorageEvalStats;
