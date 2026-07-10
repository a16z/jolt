//! Compute kernels for the modular Jolt prover: all heavy, transcript-free
//! compute behind small vocabularies, so a device backend can replace an
//! implementation without touching protocol structure. `jolt-prover` holds
//! orchestration only — every field-element crunch lives here.
//!
//! Kernel APIs consume witness oracles, field elements, and PCS setups —
//! never a transcript, never Fiat-Shamir. Sumcheck kernels implement
//! `jolt_sumcheck::ProveRounds` and are added per relation as stages demand
//! them (see `specs/clean-slate-prover.md`), in per-relation modules
//! ([`spartan_outer`] is the first). The [`NaiveSumcheckProver`] is the
//! reference tier: it interprets a relation's output `Expr` with
//! polynomial-valued leaves, making any relation provable at harness scale
//! with zero relation-specific code; optimized kernels are
//! equivalence-tested against it.
//!
//! The commitment kernel streams PCS commitments of the committed witness
//! polynomials over the proof's shared embedding grid.

mod commitment;
mod error;
mod naive;
pub mod spartan_outer;
mod sumcheck;

pub use commitment::{commit_witness, CommitmentGrid, WitnessCommitment};
pub use error::KernelError;
pub use naive::NaiveSumcheckProver;
pub use sumcheck::ProveSumcheck;
