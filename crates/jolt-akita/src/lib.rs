//! Akita PCS adapter for Jolt.
//!
//! Wraps the upstream Akita PCS over its fp128 field using Jolt's
//! `CommitmentScheme` and same-point `BatchOpeningScheme` traits.

mod backend;
mod types;

pub use backend::AkitaScheme;
pub use types::{
    AkitaBatchProof, AkitaCommitment, AkitaConfig, AkitaField, AkitaHidingCommitment,
    AkitaLayoutDigest, AkitaProverHint, AkitaProverSetup, AkitaSetupParams, AkitaSparsePolynomial,
    AkitaVerifierSetup, AKITA_D, AKITA_FIELD_MODULUS,
};
