mod api;
mod plan;
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
pub(crate) use plan::{
    outer_remainder_sequence_max_buffer_bytes_with_config,
    outer_remainder_sequence_storage_bytes_with_config,
};
pub use sequence::OuterRemainderSequence;
pub(super) use shader::SOURCE;
pub(crate) use storage::OuterRemainderSequenceStorage;
