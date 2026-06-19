//! Akita PCS adapter for Jolt.
//!
//! Wraps the LayerZero Labs Akita PCS over its fp128 field using Jolt's
//! `CommitmentScheme` and same-point `BatchOpeningScheme` traits.

mod backend;
mod layout;
mod packed;
mod types;
mod views;

pub use backend::AkitaScheme;
pub use layout::{
    PackedAdviceKind, PackedAlphabet, PackedAlphabetCounts, PackedCellAddress,
    PackedDomainCellCounts, PackedFactDomain, PackedFamily, PackedFamilyId, PackedFamilySpec,
    PackedLayoutAudit, PackedLayoutError, PackedViewKind, PackedWitnessLayout, PackedWitnessSource,
    SparsePackedWitness,
};
pub use packed::AkitaPackedScheme;
pub use types::{
    AkitaBatchProof, AkitaCommitInput, AkitaCommitment, AkitaConfig, AkitaField,
    AkitaHidingCommitment, AkitaLayoutDigest, AkitaPackedBatchProof, AkitaPackedReductionProof,
    AkitaPackedViewStatement, AkitaProverHint, AkitaProverSetup, AkitaSetupParams,
    AkitaVerifierSetup, AkitaViewFormula, AKITA_D, AKITA_FIELD_MODULUS,
};
pub use views::{
    PackedViewCatalog, PackedViewEntry, PackedViewError, PackedViewFormula, PackedViewTerm,
    PackedViewValidity,
};
