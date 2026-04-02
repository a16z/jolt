//! Schedule-driven Jolt proof verification.
//!
//! The verifier is a generic interpreter over the compiler's
//! [`VerifierSchedule`](jolt_compiler::VerifierSchedule). No per-stage
//! hand-written logic — the schedule encodes the full Fiat-Shamir replay
//! and claim-checking structure.
//!
//! No prover dependencies, no rayon, no compute backend.

pub mod config;
pub mod error;
pub mod key;
pub mod proof;
pub mod verifier;

pub use config::{OneHotConfig, OneHotParams, ProverConfig, ReadWriteConfig};
pub use error::JoltError;
pub use key::JoltVerifyingKey;
pub use proof::{JoltProof, StageProof};
pub use verifier::verify;

/// Domain separation label shared between prover and verifier transcripts.
///
/// Both sides create `Blake2bTranscript::<F>::new(TRANSCRIPT_LABEL)` independently.
/// Changing this constant is a protocol-breaking change.
pub const TRANSCRIPT_LABEL: &[u8] = b"jolt";
