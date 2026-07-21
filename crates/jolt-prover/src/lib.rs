//! Modular prover for the Jolt zkVM: a pure consumer of the
//! `SymbolicSumcheck` / `ConcreteSumcheck` / `SumcheckBatch` abstraction stack.
//!
//! `jolt-claims` defines the algebra, `jolt-verifier`'s relations and generated
//! stage drivers define the protocol structure, `jolt-sumcheck` runs the round
//! loop, and `jolt-kernels` owns every field-element crunch (including the
//! naive reference tier). This crate is orchestration only: config and
//! preprocessing, transcript sequencing, kernel invocation, typed claim
//! assembly, and proof assembly. See `specs/clean-slate-prover.md`.

mod config;
mod error;
mod preparer;
mod preprocessing;
mod prover;
pub mod stages;

pub use config::{remap_address, CommittedProgramCandidates, ProverConfig};
pub use error::ProverError;
pub use jolt_kernels::{JoltBackend, ProofSession, RetainedProgram};
pub use preparer::BackendPreparer;
pub use preprocessing::{CommittedProgramProverData, JoltProverPreprocessing};
pub use prover::prove;
