//! Akita PCS adapter for Jolt.
//!
//! Wraps the upstream Akita PCS over its fp128 field using Jolt's
//! `CommitmentScheme` trait plus Akita's backend black-box batching adapter.

mod adapters;
mod black_box_batching;
mod scheme;

pub use adapters::{
    AkitaBatchProof, AkitaCommitment, AkitaField, AkitaHidingCommitment, AkitaProverHint,
    AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup, AKITA_FIELD_MODULUS,
};
pub use black_box_batching::{
    AkitaBlackBoxBatchStatement, AkitaBlackBoxBatchWitness, AkitaBlackBoxBatching,
};
pub use scheme::AkitaScheme;
