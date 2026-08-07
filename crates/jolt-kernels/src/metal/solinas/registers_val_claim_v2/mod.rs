//! Shared register-owner model for claim reduction and value evaluation.
//!
//! `model` owns the proof-generation identity, geometry, storage accounting,
//! claim-component carrier, and opening-point order. `oracle` evaluates the
//! claim relation from CSR events and the first two value messages without a
//! dense event/index/increment staging table.
//!
//! The runtime adapter still has to bind the opaque owner identity to Metal
//! allocation identities before this module can be used by a prover backend.

pub mod model;
pub mod oracle;

pub use model::{
    RegisterClaimComponents, RegisterFamilyCarrier, RegisterFamilyGeometry,
    RegisterFamilyModelError, RegisterOwnerIdentity, RegisterOwnerSourceKind, RegisterOwnerStorage,
    RegisterValuePoint, REGISTER_ADDRESS_BITS,
};
pub use oracle::{
    claim_components_from_owner, claim_sumcheck_oracle, value_first_message_oracle,
    value_first_transition_oracle, ClaimOracleOutput, ClaimOutputValues, CubicSamples,
    OwnerOracleError, QuadraticSamples, ValueBoundRow, ValueFirstMessage, ValueFirstTransition,
};

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test setup")]
mod tests;
