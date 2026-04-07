//! Jolt zkVM prover.
//!
//! Consumes the protocol graph and produces proofs.
//! The graph says what to prove; this crate says how — backend orchestration,
//! witness building, proof serialization.

pub mod derived;
pub mod preprocessed;
pub mod preprocessing;
pub mod prove;
pub mod provider;
pub mod proving_key;
pub mod runtime;
