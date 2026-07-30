//! Modular prover for the Jolt zkVM: a pure consumer of the
//! `SymbolicSumcheck` / `ConcreteSumcheck` / `SumcheckBatch` abstraction stack.
//!
//! `jolt-claims` defines the algebra, `jolt-verifier`'s relations and generated
//! stage drivers define the protocol structure, `jolt-sumcheck` runs the round
//! loop, and `jolt-kernels` owns every field-element crunch (including the
//! naive reference tier). This crate is orchestration only: config and
//! preprocessing, transcript sequencing, kernel invocation, typed claim
//! assembly, and proof assembly. See `specs/clean-slate-prover.md`.
//!
//! Two parallel prover paths share that orchestration ([`config`],
//! [`preprocessing`], [`driver`], [`error`]):
//!
//! - `dory` — the homomorphic pipeline over an elliptic-curve PCS:
//!   streaming per-polynomial witness commitments, the stage 0–8 recipes,
//!   and the RLC-batched joint opening (`dory::prove`);
//! - `akita` — the packed pipeline over the lattice PCS: one native
//!   `OneHotTrace` commitment group, the fused-inc/reconstruction stage
//!   swaps, and the native same-point joint opening (`akita::prove`).
//!
//! Like `jolt-verifier`, one compiled prover proves exactly one protocol:
//! the `akita` feature swaps the shared wire types to the packed envelope,
//! so exactly one of the two path modules compiles into any given build.
//!
//! [`config`]: ProverConfig
//! [`preprocessing`]: JoltProverPreprocessing
//! [`driver`]: StageProver
//! [`error`]: ProverError

#[cfg(feature = "akita")]
pub mod akita;
mod config;
#[cfg(not(feature = "akita"))]
pub mod dory;
pub mod driver;
mod error;
mod preprocessing;
pub mod stages;

pub use config::{remap_address, CommittedProgramCandidates, ProverConfig};
pub use driver::{KernelSource, Proved, StageProver};
pub use error::ProverError;
pub use jolt_kernels::{JoltBackend, ProofSession};
pub use preprocessing::{CommittedProgramProverData, JoltProverPreprocessing};
