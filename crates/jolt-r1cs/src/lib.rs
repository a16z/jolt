//! R1CS constraint system types for the Jolt zkVM.
//!
//! This crate provides the pure data structures and MLE evaluation logic for
//! Rank-1 Constraint Systems. It is consumed by both the prover (jolt-spartan)
//! and verifier (jolt-verifier) but contains no proving or verification logic.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`r1cs`] | `R1CS` trait and sparse `SimpleR1CS` implementation |
//! | [`key`] | Dense Spartan key with precomputed matrix MLEs |
//! | [`uniform_key`] | Uniform key for repeated-constraint R1CS (per-cycle) |
//! | [`edge_transform`] | Combined partial evaluation for inner sumcheck |
//! | [`error`] | Error types |

pub mod edge_transform;
pub mod error;
pub mod key;
pub mod layout;
pub mod r1cs;
pub mod uniform_key;

pub use error::R1csError;
pub use key::R1csKey;
pub use layout::*;
pub use r1cs::{SimpleR1CS, R1CS};
pub use uniform_key::UniformR1csKey;
