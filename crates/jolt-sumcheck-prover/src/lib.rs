//! Universal batched sumcheck prover handler for modular Jolt.
//!
//! This crate owns the prover-side twin of [`jolt_sumcheck::BatchedSumcheckVerifier`]:
//! one canonical Fiat–Shamir loop (`prove_sumcheck`) and a small backend oracle
//! trait. Jolt stage code lowers to [`BatchedSumcheckSpec`] values; CPU and GPU
//! backends implement [`SumcheckBackend`] without reimplementing batching logic.

mod backend;
mod error;
mod handler;
mod program;
mod recorder;
mod spec;

#[cfg(test)]
mod mock;

pub use backend::SumcheckBackend;
pub use error::{BackendError, ProverError};
pub use handler::{prove_and_verify_compressed, prove_sumcheck, BatchedSumcheckProveResult};
pub use program::{ProverProgram, ProverStep, Stage};
pub use recorder::{ClearCompressedRecorder, SumcheckProofRecorder};
pub use spec::{BatchedSumcheckSpec, RoundOffset, SumcheckInstance};
