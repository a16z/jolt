//! Witness generation for the Jolt zkVM.
//!
//! Converts execution traces into multilinear polynomial evaluation tables.
//! This crate is the bridge between trace backends (RISC-V emulator, hardware
//! traces, etc.) and the proving pipeline.
//!
//! # Architecture
//!
//! - **[`TraceSource`]** — Generic input trait. Any trace backend implements
//!   this to provide rows of execution data.
//! - **[`CycleData`]** — Pre-extracted per-cycle data. The caller converts
//!   backend-specific trace rows into this flat struct.
//! - **[`WitnessConfig`]** — One-hot decomposition parameters that determine
//!   how address spaces are chunked into committed polynomials.
//! - **[`WitnessBuilder`]** — Core algorithm that processes `CycleData` rows
//!   and emits polynomial evaluation tables. Supports batch and streaming modes.
//! - **[`WitnessSink`]** — Push-based output. As witness tables are generated,
//!   chunks are pushed to the sink. The caller (jolt-zkvm) implements the sink
//!   to integrate with streaming commitment and witness storage.
//!
//! # Data flow
//!
//! ```text
//! Trace Backend          jolt-zkvm adapter         jolt-witness
//! ─────────────          ──────────────────        ────────────
//! Vec<Cycle>  ──────────→  Vec<CycleData>  ──────→  WitnessBuilder
//!                                                        │
//!                                                        ▼
//!                                                   WitnessSink
//!                                                   ┌─────────┐
//!                                                   │ Commit   │
//!                                                   │ Store    │
//!                                                   │ Both     │
//!                                                   └─────────┘
//! ```
//!
//! # Decoupling
//!
//! This crate does **not** depend on any PCS or commitment scheme. Streaming
//! commitment integration happens via the [`WitnessSink`] callback — the
//! caller decides what to do with each chunk (commit, store, both).

mod builder;
mod config;
mod cycle;
mod sink;
mod source;
pub mod trace_polys;

pub use builder::{StreamingSession, WitnessBuilder};
pub use config::WitnessConfig;
pub use cycle::CycleData;
pub use sink::{ChunkData, PolynomialKind, WitnessSink};

#[cfg(any(test, feature = "test-utils"))]
pub use sink::{CollectedPoly, CollectingSink};
pub use source::TraceSource;
pub use trace_polys::{RwEntry, TracePolynomials};

/// Re-export canonical polynomial identifiers from jolt-ir.
pub use jolt_ir::PolynomialId;
