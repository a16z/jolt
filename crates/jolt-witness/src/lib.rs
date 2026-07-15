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

// Lets derive-generated `::jolt_witness::...` paths resolve inside this
// crate's own tests.
extern crate self as jolt_witness;

pub mod backend;
#[cfg(feature = "field-inline")]
pub mod field_inline;
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;
pub mod witnesses;

mod bundle;
mod chunk;
mod consumer;
mod error;
mod shape;

pub use backend::trace::{
    JoltVmCommittedBatchStream, JoltVmCommittedStream, JoltVmWitnessConfig, JoltVmWitnessInputs,
    TraceBackend,
};
pub use backend::{BundleSource, JoltWitnessOracle};
pub use bundle::WitnessBundle;
pub use chunk::{
    PolynomialBatchChunk, PolynomialBatchStream, PolynomialChunk, PolynomialChunkKind,
    PolynomialStream,
};
pub use consumer::{stream_witnesses, CollectBundles, ConsumerSet, RowSource, StreamConsumer};
pub use error::WitnessError;
pub use shape::{PolynomialEncoding, Shape};

#[doc(hidden)]
pub mod __private {
    pub use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
    pub use jolt_program::execution::TraceRow;
}

/// XLEN of the RV64 Jolt VM this crate derives witnesses for.
pub const RV64_XLEN: usize = 64;

/// Error label for the Jolt VM witness backend.
pub(crate) const JOLT_VM_LABEL: &str = "jolt_vm";
