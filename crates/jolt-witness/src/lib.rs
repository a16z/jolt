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
//! jolt-claims'. Row sources support sequential cycle ranges and may expose
//! random-access views for parallel collection. [`JoltVmWitnessMetadata`] exposes
//! shapes and program facts without row or polynomial data; unsupported queries
//! return an unavailable-view error.

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
mod consumer;
mod error;
mod shape;

#[cfg(any(test, feature = "test-utils"))]
pub use backend::fixed::FixedBackend;
#[cfg(all(any(test, feature = "test-utils"), feature = "field-inline"))]
pub use backend::fixed::FixedFieldInline;
pub use backend::trace::{
    JoltVmWitnessConfig, JoltVmWitnessInputs, JoltVmWitnessMetadata, TraceBackend,
};
pub use backend::{
    validate_servable, BundleSource, JoltWitnessOracle, JoltWitnessPlane, ProgramSource,
};
pub use bundle::WitnessBundle;
pub use consumer::{
    collect_bundles, stream_witnesses, ChunkVisitor, CollectBundles, ConsumerSet, RandomAccessRows,
    RowSource, StreamConsumer,
};
pub use error::WitnessError;
pub use shape::{PolynomialEncoding, Shape};

#[doc(hidden)]
pub mod __private {
    pub use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
    pub use jolt_riscv::JoltTraceRow as TraceRow;
}

pub const RV64_XLEN: usize = 64;

pub(crate) const JOLT_VM_LABEL: &str = "jolt_vm";
