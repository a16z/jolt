//! Jolt zkVM prover.
//!
//! Top-level API: [`prover::prove`].
//!
//! Protocol types and claim formulas live in [`jolt_verifier::protocol`]
//! (shared by both prover and verifier). This crate re-exports them.

pub mod evaluators;
pub mod opening;
pub mod preprocessing;
pub mod proof;
pub mod prover;
pub mod r1cs;
pub mod stages;
pub mod tables;
pub mod witness;

pub use jolt_verifier::protocol;
