//! Cross-system equivalence testing between jolt-core and Bolt-generated Jolt artifacts.
//!
//! The crate exposes representation-only artifact snapshots plus focused
//! oracle, checker, tamper, and perf helpers. Protocol semantics should live in
//! Bolt, generated artifacts, kernels, or `jolt-witness`, not in this crate.

mod adapters;
mod artifacts;
pub mod bolt_oracle;
pub mod bolt_programs;
pub mod checkpoint;
pub mod checks;
pub mod commitment_oracle;
pub mod core_conversion;
pub mod core_oracle;
pub mod perf;
pub mod plan_adapters;
pub mod tamper;

pub use artifacts::{
    ArtifactSource, CommitmentArtifact, CommitmentTrace, EquivalenceRun, NamedScalar,
    OpeningBatchArtifacts, OpeningClaim, OpeningClaimKind, OpeningClaims, StageArtifacts,
    SumcheckArtifacts, TranscriptTrace, VerifierResult,
};
pub use checkpoint::{
    assert_transcripts_match, find_divergence, CheckpointTranscript, TranscriptDivergence,
    TranscriptEvent,
};
