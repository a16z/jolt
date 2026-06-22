//! Akita PCS adapter for Jolt.
//!
//! Wraps the upstream Akita PCS over its fp128 field using Jolt's
//! `CommitmentScheme` and same-point `BatchOpeningScheme` traits.

mod backend;
mod packed;
mod types;

pub use backend::AkitaScheme;
pub use packed::AkitaPackedScheme;
pub use types::{
    AkitaBatchProof, AkitaCommitInput, AkitaCommitment, AkitaConfig, AkitaField,
    AkitaHidingCommitment, AkitaLayoutDigest, AkitaPackedBatchProof, AkitaPackedReductionProof,
    AkitaPackedViewStatement, AkitaProverHint, AkitaProverSetup, AkitaSetupParams,
    AkitaVerifierSetup, AkitaViewFormula, AKITA_D, AKITA_FIELD_MODULUS,
};
