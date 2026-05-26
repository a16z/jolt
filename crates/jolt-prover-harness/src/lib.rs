//! Temporary migration harness for the modular Jolt prover.
//!
//! This crate is dev/test infrastructure. It may depend on `jolt-core` and
//! concrete backends while the modular prover is being ported. Production
//! crates must not depend on it.

pub mod checkpoint;
pub mod fixtures;
pub mod graft;
pub mod ingest;
pub mod manifest;
pub mod matrix;
pub mod metrics;
pub mod optimization;
pub mod parity;
pub mod perf;

#[cfg(feature = "field-inline")]
pub mod field_inline;

pub use checkpoint::{
    BlindFoldCheckpoint, CommitmentCheckpoint, FrontierCheckpoint, NamedValue, OpeningCheckpoint,
    StageCheckpoint,
};
pub use fixtures::{
    CoreFixtureProvider, FixtureArtifacts, FixtureKind, FixtureProvider, FixtureRequest,
    FixtureSource, StaticFixtureProvider,
};
pub use graft::{GraftPlan, GraftRecord, GraftSurface};
pub use ingest::{IngestionSurface, ProgramArtifactKind, ProverInputDescriptor};
pub use manifest::{AcceptanceMode, FeatureMode, FrontierManifest, FrontierSpec, ParityTarget};
pub use metrics::{FrontierMetrics, RunMetrics};
pub use optimization::{
    validate_frontier_optimization_ids, KnownOptimizationIds, NON_PERFORMANCE_FRONTIER_ID,
};
pub use parity::{compare_named_values, ParityMismatch, ParityReport};
pub use perf::{evaluate_perf, GateStatus, PerfEvaluation, PerfGate};

use thiserror::Error;

pub type HarnessResult<T> = Result<T, HarnessError>;

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("invalid frontier manifest entry `{frontier}`: {reason}")]
    InvalidManifest {
        frontier: &'static str,
        reason: String,
    },
    #[error("frontier `{frontier}` does not support acceptance mode `{mode:?}`")]
    InvalidAcceptanceMode {
        frontier: &'static str,
        mode: AcceptanceMode,
    },
    #[error("fixture `{fixture:?}` is not available for `{context}`")]
    FixtureUnavailable {
        fixture: FixtureKind,
        context: &'static str,
    },
    #[error("invalid prover input from `{surface}`: {reason}")]
    InvalidIngestion { surface: String, reason: String },
    #[error("feature `{feature}` is required for `{context}`")]
    MissingFeature {
        feature: &'static str,
        context: &'static str,
    },
    #[error("invalid optimization inventory: {reason}")]
    InvalidOptimizationInventory { reason: String },
    #[error("parity target `{target}` failed: {reason}")]
    Parity { target: String, reason: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
