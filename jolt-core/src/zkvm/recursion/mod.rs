//! Recursion SNARK implementation for Dory commitment scheme
//!
//! This module implements a two-stage SNARK protocol for proving recursion constraints
//! that arise from the Dory polynomial commitment scheme.
//!
//! ## Protocol Overview
//!
//! The recursion SNARK consists of two stages:
//!
//! ### Stage 1: Constraint Sumchecks (Parallel)
//! Three independent sumcheck instances prove different constraint types:
//! - **GT Exponentiation**: Proves constraints for pairing group exponentiation
//! - **GT Multiplication**: Proves constraints for pairing group multiplication
//! - **G1 Scalar Multiplication**: Proves constraints for elliptic curve scalar multiplication
//!
//! ### Stage 2: Virtualization Sumcheck
//! A single sumcheck that virtualizes all Stage 1 claims into a unified proof
//!
//! ## Module Structure
//! - `constraints_sys`: Constraint system management and matrix building
//! - `stage1/`: Stage 1 constraint sumcheck implementations
//! - `stage2/`: Stage 2 virtualization sumcheck
//! - `utils/`: Shared utilities and helpers
//! - `recursion_prover`: Unified prover orchestrating both stages
//! - `recursion_verifier`: Unified verifier for the complete protocol

pub mod constraints_sys;
pub mod recursion_prover;
pub mod recursion_verifier;
pub mod stage1;
pub mod stage2;
pub mod utils;
pub mod witness;

#[cfg(test)]
mod tests;

// Re-export main types
pub use constraints_sys::{ConstraintSystem, ConstraintType, DoryMatrixBuilder, PolyType};
pub use recursion_prover::{RecursionProof, RecursionProver};
pub use recursion_verifier::{RecursionVerifier, RecursionVerifierInput};
pub use stage1::{
    g1_scalar_mul::{G1ScalarMulProver, G1ScalarMulVerifier},
    gt_mul::{GtMulProver, GtMulVerifier},
    square_and_multiply::{SquareAndMultiplyProver, SquareAndMultiplyVerifier},
};
pub use stage2::virtualization::{
    RecursionVirtualizationParams, RecursionVirtualizationProver, RecursionVirtualizationVerifier,
};
pub use witness::{DoryRecursionWitness, G1ScalarMulWitness, GTExpWitness, GTMulWitness, WitnessData};