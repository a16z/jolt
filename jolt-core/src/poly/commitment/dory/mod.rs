//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.

mod commitment_scheme;
mod dory_globals;
mod jolt_dory_routines;
pub mod recursion;
mod wrappers;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod recursion_test;

pub use commitment_scheme::DoryCommitmentScheme;
pub use dory_globals::{DoryContext, DoryGlobals};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, BN254,
};
pub use recursion::{JoltWitness, JoltWitnessGenerator};
