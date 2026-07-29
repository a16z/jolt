//! Dory polynomial commitment scheme
//!
//! This module adapts Jolt's polynomial commitment traits to final-dory's
//! arkworks backend.

mod commitment_scheme;
mod dory_globals;
mod jolt_dory_routines;
#[cfg(not(target_arch = "wasm32"))]
mod urs_lock;
mod wrappers;

#[cfg(test)]
mod tests;

#[cfg(feature = "zk")]
pub use commitment_scheme::bind_opening_inputs_zk;
pub use commitment_scheme::{bind_opening_inputs, DoryCommitmentScheme, DoryOpeningProofHint};
pub use dory_globals::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, BN254,
};
