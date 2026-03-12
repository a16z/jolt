//! Jolt zkVM prover.
//!
//! Top-level API: [`prover::prove`] and [`prover::verify`].
//!
//! Composes all Jolt sub-crates into a complete proving system for
//! RISC-V (RV64IMAC) execution traces. See `crates/zkvm_spec.md`
//! for the full design specification.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`prover`] | Top-level [`prove`](prover::prove) and [`verify`](prover::verify) API |
//! | [`preprocessing`] | Circuit key construction and PCS setup |
//! | [`r1cs`] | Jolt R1CS constraints → [`UniformSpartanKey`](jolt_spartan::UniformSpartanKey) |
//! | [`witness`] | Witness generation: trace → R1CS witnesses + committed polynomials |
//! | [`stage`] | [`ProverStage`](stage::ProverStage) trait + [`StageBatch`](stage::StageBatch) |
//! | [`pipeline`] | [`prove_stages`](pipeline::prove_stages) — prover pipeline driver |
//! | [`evaluators`] | [`SumcheckCompute`](jolt_sumcheck::SumcheckCompute) implementations |
//! | [`stages`] | [`ProverStage`](stage::ProverStage) implementations |
//!
//! Claim definitions and polynomial tags live in [`jolt_ir::zkvm`] — the
//! single source of truth shared by prover and verifier.

pub mod evaluators;
pub mod pipeline;
pub mod preprocessing;
pub mod proof;
pub mod prover;
pub mod r1cs;
pub mod stage;
pub mod stages;
pub mod witness;
