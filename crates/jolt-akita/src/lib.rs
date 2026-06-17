//! Akita PCS adapter for Jolt.
//!
//! Wraps the LayerZero Labs Akita PCS over its fp128 field using Jolt's
//! `CommitmentScheme` and same-point `BatchOpeningScheme` traits.

mod backend;
mod layout;
mod types;

pub use backend::AkitaScheme;
pub use layout::{
    PackedAdviceKind, PackedAlphabet, PackedAlphabetCounts, PackedCellAddress,
    PackedDomainCellCounts, PackedFactDomain, PackedFamily, PackedFamilyId, PackedFamilySpec,
    PackedLayoutAudit, PackedLayoutError, PackedViewKind, PackedWitnessLayout, PackedWitnessSource,
    SparsePackedWitness,
};
pub use types::{
    AkitaBatchProof, AkitaCommitInput, AkitaCommitment, AkitaConfig, AkitaField,
    AkitaHidingCommitment, AkitaLayoutDigest, AkitaPackedViewStatement, AkitaProverHint,
    AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup, AkitaViewFormula, AKITA_D,
};
