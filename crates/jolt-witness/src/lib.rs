//! The trace→witness transformation layer for Jolt proving.
//!
//! Three maps, three homes:
//!
//! ```text
//! trace rows ──(one-to-many: Extract impls)──▶ atomic witnesses (witnesses/)
//! atomic witnesses ──(many-to-many: bundles)──▶ consumer bundles
//! bundles / ids ──(backends)──▶ kernels & commitment
//! ```
//!
//! A witness is an atomic value newtype with a single-sourced derivation from
//! a trace row. Backends serve them two ways: the object-safe id-indexed
//! [`JoltWitnessOracle`] (the naive interpreter's path — one exhaustive match
//! over jolt-claims ids, no wildcard) and typed bundles over the streaming
//! pass. This crate defines **no id vocabulary of its own** — all ids are
//! jolt-claims'. Every public contract is sequential over cycle ranges;
//! random access to trace rows is deliberately inexpressible, so a
//! checkpointed, re-emulating trace source can implement every signature
//! honestly.

pub mod backend;
#[cfg(feature = "field-inline")]
pub mod field_inline;
pub mod witnesses;

mod chunk;
mod error;
mod shape;

pub use backend::trace::{
    JoltVmCommittedBatchStream, JoltVmCommittedStream, JoltVmStage5InstructionReadRafRows,
    JoltVmStage6Row, JoltVmStage6Rows, JoltVmWitnessConfig, JoltVmWitnessInputs,
    Stage5InstructionReadRafRow, TraceBackend,
};
pub use backend::JoltWitnessOracle;
pub use chunk::{
    PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk, PolynomialChunkKind,
    PolynomialStream,
};
pub use error::WitnessError;
pub use shape::{PolynomialEncoding, Shape, WitnessDimensions};

/// XLEN of the RV64 Jolt VM this crate derives witnesses for.
pub const RV64_XLEN: usize = 64;

/// Error label for the Jolt VM witness backend.
pub(crate) const JOLT_VM_LABEL: &str = "jolt_vm";
