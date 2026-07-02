//! Akita PCS adapter for Jolt.
//!
//! Wraps the upstream Akita PCS over its fp128 field using Jolt's
//! `CommitmentScheme` trait plus an adapter delegating same-point batches to
//! Akita's native batched opening protocol.

mod adapters;
mod native_batching;
mod scheme;

pub use adapters::{
    AkitaBackendFlavor, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHidingCommitment,
    AkitaProverHint, AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup, AKITA_ONE_HOT_K,
};
pub use native_batching::{
    AkitaNativeBatchStatement, AkitaNativeBatchWitness, AkitaNativeBatching,
};
pub use scheme::AkitaScheme;

/// Jolt↔Akita basis-order bridging, exposed so benchmarks measuring the raw
/// backend use the exact transform the adapter uses.
#[doc(hidden)]
pub use adapters::{jolt_to_akita_evals, jolt_to_akita_index, reverse_point};
