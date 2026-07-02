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
    AkitaProverHint, AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup, AKITA_FIELD_MODULUS,
};
pub use native_batching::{
    AkitaNativeBatchStatement, AkitaNativeBatchWitness, AkitaNativeBatching,
};
pub use scheme::AkitaScheme;
