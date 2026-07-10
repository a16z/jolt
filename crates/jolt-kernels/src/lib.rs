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
//! implementation. The [`NaiveSumcheckProver`] is the reference tier: it
//! interprets a relation's output `Expr` with polynomial-valued leaves,
//! making any relation whose leaves are multilinear provable at harness
//! scale with zero relation-specific code; optimized kernels are
//! equivalence-tested against it. Derived ids correspond one-to-one with
//! multilinears by design; the single enumerated escape so far is the
//! Spartan outer remainder's post-uni-skip stream round ([`spartan_outer`],
//! whose expanded coefficient leaves are quadratic in the stream variable —
//! a uni-skip structure, not a fixable factoring). See
//! `specs/clean-slate-prover.md`, "The backend seam".
//!
//! The commitment kernel streams PCS commitments of the committed witness
//! polynomials over the proof's shared embedding grid.

mod backend;
mod commitment;
mod error;
pub mod instruction_claim_reduction;
mod naive;
pub mod ram_output_check;
pub mod ram_raf_evaluation;
pub mod ram_read_write;
pub mod spartan_outer;
pub mod spartan_product;
mod sumcheck;
mod views;

pub use backend::{JoltBackend, ProofSession, ReferenceBackend};
pub use commitment::{CommitWitness, CommitmentGrid, WitnessCommitment};
pub use error::KernelError;
pub use naive::NaiveSumcheckProver;
pub use sumcheck::ProveSumcheck;
