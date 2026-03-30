//! Jolt zkVM prover.
//!
//! Consumes the protocol graph and produces proofs.
//! The graph says what to prove; this crate says how — backend orchestration,
//! witness building, proof serialization.

pub mod preprocessing;
pub mod proof;
pub mod prover;
pub mod r1cs;
pub mod runtime;
