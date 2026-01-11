//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.

mod commitment_scheme;
mod dory_globals;
pub mod g1_scalar_mul_witness;
pub mod gt_mul_witness;
mod jolt_dory_routines;
pub mod recursion;
pub mod wrappers;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod recursion_test;

pub use commitment_scheme::DoryCommitmentScheme;
pub use dory_globals::{DoryContext, DoryGlobals};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use recursion::{JoltWitness, JoltWitnessGenerator};
pub use wrappers::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, BN254,
};
