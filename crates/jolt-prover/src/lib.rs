//! Protocol orchestration for modular Jolt proving.
//!
//! This crate owns proof order, transcript scheduling, and typed verifier-proof
//! construction. Concrete compute implementations live in `jolt-backends`.

mod assembly;
mod builder;
#[cfg(feature = "zk")]
mod committed;
mod config;
mod error;
mod preprocessing;
mod prover;
pub mod stages;
#[cfg(feature = "frontier-harness")]
mod timing;
mod transcript;

pub use config::{ProverConfig, ProverFeatureSet, ProverProofShape};
pub use error::ProverError;
pub use preprocessing::JoltProverPreprocessing;
pub use prover::{prove, prove_with_output, BlindFoldProverBackend, ProverOutput};
#[cfg(feature = "frontier-harness")]
pub use timing::{reset_stage_timings, take_stage_timings, StageTiming};
pub use transcript::{
    absorb_stage0_transcript, initialize_proof_transcript, Stage0TranscriptContext,
};
