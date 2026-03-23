//! Jolt zkVM prover.
//!
//! Consumes the protocol graph from [`jolt_ir`] and produces proofs.
//! The graph says what to prove; this crate says how — evaluator dispatch,
//! backend orchestration, witness building, proof serialization.

pub mod evaluators;
pub mod preprocessing;
pub mod proof;
pub mod prover;
pub mod r1cs;
pub mod tables;
pub mod witness;
pub mod witness_builder;
