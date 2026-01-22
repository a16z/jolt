//! Stage 5: Jagged assist sumcheck
//!
//! Batch MLE verification for the jagged transform.

pub mod jagged_assist;

pub use jagged_assist::{
    JaggedAssistEvalPoint, JaggedAssistParams, JaggedAssistProof, JaggedAssistProver,
    JaggedAssistVerifier,
};
