//! Witness data pipeline interface — polynomial materialization + lookup trace.
//!
//! The runtime calls [`BufferProvider::materialize`] whenever it needs
//! polynomial data for a compute op (to upload to a backend) or a PCS op
//! (to feed the commitment scheme host-side). The provider is
//! backend-agnostic: it returns raw field data and the runtime decides how
//! to consume it.
//!
//! [`LookupTraceData`] carries per-cycle non-field trace data (u128 keys,
//! table indices, flags) that can't be expressed as polynomial evaluations
//! but is needed by handlers that orchestrate lookup-protocol ops.
//!
//! # Why in jolt-compiler
//!
//! The compiler defines the IR (ops, polynomials, challenges); it also
//! defines the shape of witness data the runtime must supply to execute
//! that IR. Both the runtime (`jolt-zkvm`, consumer) and the witness
//! providers (`jolt-witness`, implementors) depend on `jolt-compiler`, so
//! the trait lives here as part of the IR contract.

use crate::PolynomialId;
use jolt_field::Field;

/// Per-cycle trace data for the instruction lookup sumcheck.
///
/// Provided alongside polynomial data as an input to the prover runtime.
/// Non-field fields (u128 keys, Option<usize>, bool) that can't be expressed
/// as polynomial evaluations. Consumed by the protocol-specific lookup
/// handlers (suffix scatter, Q-buffer scatter, instance-weight updates,
/// RA materialization, combined-val materialization).
pub struct LookupTraceData {
    /// Per-cycle lookup key (128-bit packed), T entries.
    pub lookup_keys: Vec<u128>,
    /// Per-cycle lookup table index, T entries. None for cycles with no lookup.
    pub table_kind_indices: Vec<Option<usize>>,
    /// Per-cycle interleaved-operands flag, T entries.
    pub is_interleaved: Vec<bool>,
}

/// Materializes polynomial data for the prover runtime.
///
/// The runtime calls [`materialize`](BufferProvider::materialize) whenever it
/// needs polynomial data — for device upload (compute ops) or direct host
/// access (PCS ops). The provider is backend-agnostic: it returns raw field
/// data and the runtime decides how to consume it.
///
/// Returns [`Cow::Borrowed`] for stored polynomials (zero-copy) and
/// [`Cow::Owned`] for computed polynomials (R1CS, virtual).
pub trait BufferProvider<F: Field> {
    /// Materialize polynomial data for the given ID.
    fn materialize(&self, poly_id: PolynomialId) -> std::borrow::Cow<'_, [F]>;

    /// Release stored polynomial data to reclaim memory.
    fn release(&mut self, _poly_id: PolynomialId) {}

    /// Per-cycle lookup trace data used by instruction-lookup handlers.
    /// Returns `None` when no lookup sumcheck is in the schedule.
    fn lookup_trace(&self) -> Option<&LookupTraceData> {
        None
    }
}
