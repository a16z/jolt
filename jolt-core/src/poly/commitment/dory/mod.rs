//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.

mod commitment_scheme;
mod dory_globals;
mod jolt_dory_routines;
mod wrappers;

#[cfg(test)]
mod tests;

#[cfg(feature = "zk")]
pub use commitment_scheme::bind_opening_inputs_zk;
pub use commitment_scheme::{bind_opening_inputs, DoryCommitmentScheme};
pub use dory_globals::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, BN254,
};
