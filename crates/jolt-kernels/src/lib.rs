//! Compute kernels for the modular Jolt prover: all heavy, transcript-free
//! compute behind the backend seam, so implementations swap without touching
//! protocol structure. `jolt-prover` holds orchestration only — every
//! field-element crunch lives here, behind a slot of the [`JoltBackend`]
//! registry it proves against.
//!
//! Kernel APIs consume witness oracles, field elements, and PCS setups —
//! never a transcript, never Fiat-Shamir — and return canonical values.
//! Sumcheck kernels implement `jolt_sumcheck::ProveRounds` and are added per
//! relation as stages demand them, each in a per-relation module defining
//! the slot's object-safe factory/instance traits next to the reference
//! implementation ([`spartan_outer`] is the first). The
//! [`NaiveSumcheckProver`] is the reference tier: it interprets a relation's
//! output `Expr` with polynomial-valued leaves, making any relation provable
//! at harness scale with zero relation-specific code; optimized kernels are
//! equivalence-tested against it. See `specs/clean-slate-prover.md`,
//! "The backend seam".
//!
//! The commitment kernel streams PCS commitments of the committed witness
//! polynomials over the proof's shared embedding grid.

mod backend;
mod commitment;
mod error;
mod naive;
pub mod spartan_outer;
mod sumcheck;

pub use backend::{JoltBackend, ProofSession, ReferenceBackend};
pub use commitment::{commit_witness, CommitWitness, CommitmentGrid, WitnessCommitment};
pub use error::KernelError;
pub use naive::NaiveSumcheckProver;
pub use sumcheck::ProveSumcheck;
