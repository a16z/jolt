//! Isolated successor design for the RAM RAF-evaluation member.
//!
//! This packet is not registered with the Metal backend. It freezes the compact
//! producer ABI, exact cost model, independent scalar oracles, and candidate
//! shader entry points before shared runtime work begins.

pub mod abi;
pub mod model;
pub mod oracle;

pub use abi::{
    build_bucket_projection, validate_access_records, validate_bucket_projection,
    RamRafAccessRecord, RamRafBucketDescriptor, RamRafBucketProjection, RamRafBucketRecord,
    RamRafBucketedParams, RamRafCompactError, RamRafDirectParams, RamRafFinalizeParams,
    RamRafStatus,
};
pub use model::{
    execution_screen, Geometry, HostSparseProjection, KnownRoof, ModelError, RamRafExecutionLane,
    RamRafTopology, RoofProjection, StoragePlan, TARGET_FIBONACCI_OBSERVED_NONZERO_SUBTOTALS,
    TARGET_FIBONACCI_TOPOLOGY,
};
pub use oracle::{
    bucket_pushforward_oracle, compact_pushforward_oracle, dense_pushforward_oracle,
    direct_equality, prove_affine_address_rounds, AffineProofOutput, OracleError,
    QuadraticEvaluations,
};

pub const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "isolated packet tests fail on malformed fixtures"
)]
mod tests;
