//! Groth16 Transpilation for Stage 1 Verification
//!
//! This module provides Groth16 circuits for verifying Jolt's Stage 1 proof,
//! enabling EVM-efficient on-chain verification.
//!
//! ## Available Implementations
//!
//! - **arkworks**: Native Rust implementation using the arkworks ecosystem
//!   - Mature, well-audited crates
//!   - Direct integration with Jolt's existing ark-bn254 dependencies
//!   - Enable with `groth16-stable` or `groth16-git` features
//!
//! Future implementations may include:
//! - **circom**: DSL-based approach with snarkjs tooling
//! - **halo2**: PLONKish arithmetization (no trusted setup)
//! - **sp1**: zkVM-based recursive verification

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub mod arkworks;

// Re-export arkworks types at the groth16 level for backwards compatibility
#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub use arkworks::circuit::{Stage1Circuit, Stage1CircuitConfig};

#[cfg(any(feature = "groth16-stable", feature = "groth16-git"))]
pub use arkworks::witness::Stage1CircuitData;
