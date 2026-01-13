//! Stage 3: Jagged transform sumcheck
//!
//! This stage reduces claims on the sparse constraint matrix M(s,x)
//! to claims on a dense polynomial q(i) that excludes zero entries.
//!
//! Sub-modules:
//! - `jagged`: Main Stage 3 sumcheck (sparse → dense transform)
//! - `jagged_assist`: Stage 3b - Batch MLE verification optimization

pub mod branching_program;
pub mod jagged;
pub mod jagged_assist;

pub use jagged::{JaggedSumcheckParams, JaggedSumcheckProver, JaggedSumcheckVerifier};
pub use jagged_assist::{
    JaggedAssistEvalPoint, JaggedAssistParams, JaggedAssistProof, JaggedAssistProver,
    JaggedAssistVerifier,
};
