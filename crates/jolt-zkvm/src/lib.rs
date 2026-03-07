//! Top-level zkVM prover and verifier orchestration.
//!
//! Composes all Jolt sub-crates into a complete proving system for
//! RISC-V (RV64IMAC) execution traces. See `crates/zkvm_spec.md`
//! for the full design specification.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`claims`] | IR-based claim definitions (single source of truth) |
//! | [`tags`] | Opaque polynomial and sumcheck identity tags |
//! | [`witness`] | [`WitnessStore`](witness::WitnessStore) — polynomial evaluation table storage |
//! | [`stage`] | [`ProverStage`](stage::ProverStage) trait + [`StageBatch`](stage::StageBatch) |
//! | [`pipeline`] | [`prove_stages`](pipeline::prove_stages) — prover pipeline driver |
//! | [`witnesses`] | [`SumcheckCompute`](jolt_sumcheck::SumcheckCompute) implementations |
//! | [`stages`] | [`ProverStage`](stage::ProverStage) implementations |

pub mod claims;
pub mod pipeline;
pub mod stage;
pub mod stages;
pub mod tags;
pub mod witness;
pub mod witnesses;
