//! Jolt zkVM prover.
//!
//! Consumes the protocol graph and produces proofs.
//! The graph says what to prove; this crate says how — backend orchestration,
//! witness building, proof serialization.

pub mod preprocessing;
pub mod prove;
pub mod proving_key;
pub mod runtime;
pub mod scalar_expr;
