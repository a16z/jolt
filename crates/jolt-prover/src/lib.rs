//! Modular prover for the Jolt zkVM: a pure consumer of the
//! `SymbolicSumcheck` / `ConcreteSumcheck` / `SumcheckBatch` abstraction stack.
//!
//! `jolt-claims` defines the algebra, `jolt-verifier`'s relations and generated
//! stage drivers define the protocol structure, and this crate adds exactly two
//! things: polynomial data and a round loop (the engine itself lives in
//! `jolt-sumcheck`). See `specs/clean-slate-prover.md`.
//!
//! The [`NaiveSumcheckProver`] is the semantic ground truth: it interprets a
//! relation's output `Expr` with polynomial-valued leaves, making any relation
//! provable at harness scale with zero kernel code. It is a test oracle, never
//! a performance path — optimized kernels are equivalence-tested against it.

mod config;
mod error;
mod naive;
mod preprocessing;
pub mod stages;
mod sumcheck;

pub use config::{remap_address, ProverConfig};
pub use error::ProverError;
pub use naive::NaiveSumcheckProver;
pub use preprocessing::JoltProverPreprocessing;
pub use sumcheck::ProveSumcheck;
