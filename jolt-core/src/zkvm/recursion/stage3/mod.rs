//! Stage 3: Jagged transform sumcheck
//!
//! This stage reduces claims on the sparse constraint matrix M(s,x)
//! to claims on a dense polynomial q(i) that excludes zero entries.

pub mod branching_program;
pub mod jagged;

pub use jagged::{JaggedSumcheckParams, JaggedSumcheckProver, JaggedSumcheckVerifier};
