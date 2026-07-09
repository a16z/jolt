//! Compute kernels for the modular Jolt prover: heavy, transcript-free
//! compute behind a small vocabulary, so a device backend can replace an
//! implementation without touching protocol structure.
//!
//! Kernel APIs take and return field elements (and commitments) only — no
//! transcript, no Fiat-Shamir. Sumcheck kernels, when they arrive, implement
//! `jolt_sumcheck::ProveRounds` directly and are equivalence-tested against
//! the naive `Expr` interpreter in `jolt-prover`; they are added per relation
//! as stages demand them (see `specs/clean-slate-prover.md`).
//!
//! The first resident is the witness-commitment kernel: streaming PCS
//! commitment of the committed witness polynomials over the proof's shared
//! embedding grid.

mod commitment;
mod error;

pub use commitment::{commit_witness, CommitmentGrid, WitnessCommitment};
pub use error::KernelError;
