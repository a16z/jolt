//! Polynomial descriptors: operational semantics for polynomial identities.
//!
//! Every polynomial in the protocol carries a [`PolynomialDescriptor`] that
//! specifies its data source, storage representation, commitment status, and
//! witness slot. This is the single source of truth that downstream crates
//! use for generic dispatch — no hardcoded identity matching required.

use serde::{Deserialize, Serialize};

/// Operational semantics for a polynomial identity.
///
/// Returned by [`PolynomialId::descriptor`] and used by the runtime,
/// buffer providers, and witness layer to dispatch generically on polynomial
/// kind rather than matching specific identities.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PolynomialDescriptor {
    /// How this polynomial gets its data.
    pub source: PolySource,
    /// Whether this polynomial has a PCS commitment.
    pub committed: bool,
    /// Storage representation hint for the buffer layer.
    pub storage: StorageHint,
    /// Which slot of per-cycle input data this polynomial reads from.
    /// `None` for polynomials not derived from the execution trace
    /// (R1CS, derived, preprocessed, or separately-inserted like advice).
    pub witness_slot: Option<WitnessSlot>,
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
    /// A single witness variable column: `witness[c * V_pad + var_index]`
    /// for each cycle `c`. Used to evaluate individual R1CS input polynomials
    /// at the sumcheck challenge point.
    Variable(usize),
}

/// Storage representation hint for the buffer layer.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum StorageHint {
    /// Full dense evaluation buffer (`Vec<F>`).
    Dense,
    /// One-hot decomposition: sparse index per cycle, expanded at finish.
    OneHot,
    /// Never fully materialized on host — computed on-demand per load.
    OnDemand,
}

/// Identifies which slot of per-cycle input data a witness polynomial reads from.
///
/// The witness layer uses this to generically populate polynomial buffers
/// from indexed trace data — no hardcoded polynomial identity matching in
/// `Polynomials::push`.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum WitnessSlot {
    /// Read from `CycleInput::dense[index]` as i128, convert to field element.
    Dense(usize),
    /// Read one-hot chunk `dim` from `CycleInput::one_hot[source]`.
    /// Chunk extraction uses `PolynomialConfig::chunk()` at materialization time.
    OneHotChunk { source: usize, dim: usize },
}

impl WitnessSlot {
    /// Number of dense witness slots in per-cycle input data.
    pub const NUM_DENSE: usize = 3;
    /// Number of one-hot source values in per-cycle input data.
    pub const NUM_ONE_HOT: usize = 4;

    /// Dense slot: register write increment (`rd post − rd pre`).
    pub const RD_INC: usize = 0;
    /// Dense slot: RAM write increment (`ram post − ram pre`).
    pub const RAM_INC: usize = 1;
    /// Dense slot: BN254 Fr coprocessor register-file write increment (one
    /// 256-bit element per cycle; stored as the committed `FieldRegInc`
    /// polynomial, padded from log_t to LOG_K_FR + log_t at materialize
    /// time).
    pub const FIELD_REG_INC: usize = 2;

    /// One-hot source: instruction lookup (128-bit interleaved encoding).
    pub const INSTRUCTION: usize = 0;
    /// One-hot source: bytecode PC.
    pub const BYTECODE: usize = 1;
    /// One-hot source: remapped RAM address.
    pub const RAM: usize = 2;
    /// One-hot source: BN254 Fr coprocessor write address (`frd & 0xF`,
    /// or sentinel 255 on non-FR cycles).
    pub const FIELD_REG: usize = 3;
}
