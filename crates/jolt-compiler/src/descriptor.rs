//! Polynomial descriptors: operational semantics for polynomial identities.
//!
//! Every polynomial in the protocol carries a [`PolynomialDescriptor`] that
//! specifies its data source, storage representation, and commitment status.
//! This is the single source of truth that downstream crates use for generic
//! dispatch — no hardcoded identity matching required.

use serde::{Deserialize, Serialize};

/// Operational semantics for a polynomial identity.
///
/// Returned by [`PolynomialSpec::descriptor`](super::PolynomialSpec) and used
/// by the runtime, buffer providers, and verifier to dispatch generically
/// on polynomial kind rather than matching specific identities.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PolynomialDescriptor {
    /// How this polynomial gets its data.
    pub source: PolySource,
    /// Whether this polynomial has a PCS commitment.
    pub committed: bool,
    /// Storage representation hint for the buffer layer.
    pub storage: StorageHint,
}

/// How a polynomial's evaluation data is produced.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PolySource {
    /// Materialized from execution trace during witness generation.
    Witness,
    /// Computed on-demand from R1CS constraint matrices and witness.
    R1cs(R1csColumn),
    /// Derived during protocol execution (eq tables, intermediate products).
    Derived,
    /// Loaded from preprocessed/verifying key data.
    Preprocessed,
}

/// Which column of the R1CS relation `Az ∘ Bz = Cz`.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum R1csColumn {
    Az,
    Bz,
    Cz,
    CombinedRow,
}

/// Storage representation hint for the buffer layer.
///
/// Downstream crates (e.g. `jolt-witness`) use this to decide allocation
/// strategy. The actual chunk size `k` for one-hot polynomials comes from
/// [`PolynomialConfig`](jolt_witness) at materialization time.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum StorageHint {
    /// Full dense evaluation buffer (`Vec<F>`).
    Dense,
    /// One-hot decomposition: sparse index per cycle, expanded at finish.
    /// The chunk size `k` is resolved from `PolynomialConfig`.
    OneHot,
    /// Never fully materialized on host — computed on-demand per load.
    OnDemand,
}
