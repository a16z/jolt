//! Recursion SNARK implementation for Dory commitment scheme
//!
//! This module implements a three-stage SNARK protocol with optimizations for proving
//! recursion constraints that arise from the Dory polynomial commitment scheme.
//!
//! ## Protocol Overview
//!
//! The recursion SNARK consists of three stages plus optimizations:
//!
//! ### Stage 1: Constraint Sumchecks (Parallel)
//! Three independent sumcheck instances prove different constraint types:
//! - **GT Exponentiation**: Proves constraints for pairing group exponentiation
//! - **GT Multiplication**: Proves constraints for pairing group multiplication
//! - **G1 Scalar Multiplication**: Proves constraints for elliptic curve scalar multiplication
//!
//! ### Stage 2: Direct Evaluation Protocol
//! Verifies virtual polynomial claims from Stage 1 using direct evaluation instead
//! of sumcheck, leveraging the special structure of the constraint matrix
//!
//! ### Stage 3: Sparse Polynomial Opening
//! Uses sumcheck to open the sparse constraint matrix at a random point
//!
//! ### Stage 3b: Jagged Assist (Optimization)
//! Batch verification protocol that reduces verifier cost for evaluating multiple
//! polynomial openings from O(K × bits) to O(bits) operations
//!
//! ## Module Structure
//! - `constraints_sys`: Constraint system management and matrix building
//! - `stage1/`: Stage 1 constraint sumcheck implementations
//! - `stage2/`: Stage 2 virtualization sumcheck
//! - `utils/`: Shared utilities and helpers
//! - `recursion_prover`: Unified prover orchestrating both stages
//! - `recursion_verifier`: Unified verifier for the complete protocol

pub mod bijection;
pub mod constraints_sys;
pub mod recursion_prover;
pub mod recursion_verifier;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod utils;
pub mod witness;

#[cfg(test)]
mod tests;

// Re-export main types
pub use bijection::{ConstraintMapping, JaggedTransform, VarCountJaggedBijection};
pub use constraints_sys::{
    ConstraintSystem, ConstraintType, DoryMatrixBuilder, PolyType, RecursionMetadataBuilder,
};
pub use recursion_prover::{RecursionProof, RecursionProver};
pub use recursion_verifier::{RecursionVerifier, RecursionVerifierInput};
pub use stage1::{
    g1_scalar_mul::{G1ScalarMulProver, G1ScalarMulVerifier},
    gt_mul::{GtMulProver, GtMulVerifier},
};
pub use stage2::virtualization::{
    extract_virtual_claims_from_accumulator, DirectEvaluationParams, DirectEvaluationProver,
    DirectEvaluationVerifier,
};
pub use stage3::jagged::{JaggedSumcheckParams, JaggedSumcheckProver, JaggedSumcheckVerifier};
pub use witness::{
    DoryRecursionWitness, G1ScalarMulWitness, GTExpWitness, GTMulWitness, WitnessData,
};
