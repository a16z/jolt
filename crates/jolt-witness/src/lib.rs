//! Witness generation for the Jolt zkVM.
//!
//! Converts execution traces into multilinear polynomial evaluation tables
//! and R1CS witness vectors. This crate is the bridge between trace backends
//! (RISC-V emulator, hardware traces, etc.) and the proving pipeline.
//!
//! # Architecture
//!
//! - **[`bytecode`]** — Bytecode preprocessing for PC expansion.
//! - **[`cycle_data`]** — Converts tracer `Cycle` → `CycleData`.
//! - **[`flags`]** — Maps tracer instructions to jolt-instructions flag arrays.
//! - **[`r1cs_inputs`]** — Converts `Cycle` → R1CS per-cycle witness vector.
//! - **[`WitnessBuilder`]** — Core algorithm that processes `CycleData` rows
//!   and emits polynomial evaluation tables.
//! - **[`Witness`]** — Owns evaluation tables, implements [`WitnessSink`] for
//!   direct ingestion from the builder, and provides [`WitnessProvider`] for
//!   runtime buffer loading.
//! - **[`generate_witnesses`]** — End-to-end orchestration from trace to all
//!   witness outputs.

mod builder;
pub mod bytecode;
mod config;
pub mod cycle;
pub mod cycle_data;
pub mod flags;
pub mod generate;
pub mod polynomial_id;
pub mod r1cs_inputs;
mod sink;
pub mod trace_polys;
mod witness;

pub use builder::{StreamingSession, WitnessBuilder};
pub use config::WitnessConfig;
pub use cycle::CycleData;
pub use generate::{generate_witnesses, WitnessOutput};
pub use sink::{ChunkData, PolynomialKind, WitnessSink};
pub use witness::{Witness, WitnessProvider};

#[cfg(any(test, feature = "test-utils"))]
pub use sink::{CollectedPoly, CollectingSink};
pub use trace_polys::{RwEntry, TracePolynomials};

pub use polynomial_id::PolynomialId;
