//! Jolt protocol definitions shared by prover and verifier.
//!
//! - [`types`] — Evaluation data that flows between stages
//! - [`claims`] — Input claim formulas (mathematical identities)

pub mod claims;
pub mod types;

pub use types::*;
