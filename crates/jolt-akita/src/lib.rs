//! Akita PCS adapter shell.
//!
//! This crate currently exposes a transparent mock backend that normalizes and
//! binds Akita-shaped batch opening statements. It is a protocol-facing scaffold
//! for the real lattice backend, not a cryptographic Akita implementation.

mod backend;
mod field;
mod types;

pub use backend::AkitaScheme;
pub use field::{to_akita_claim, AkitaClaimField};
pub use types::{
    AkitaBatchProof, AkitaCommitInput, AkitaCommitment, AkitaFieldMode, AkitaHidingCommitment,
    AkitaLayoutDigest, AkitaPackedViewStatement, AkitaProverHint, AkitaProverSetup, AkitaSetup,
    AkitaSetupKey, AkitaSetupMode, AkitaSetupParams, AkitaVerifierSetup, AkitaViewFormula,
};
