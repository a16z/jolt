//! ZkLean Extractor Library
//!
//! Provides the MleAst symbolic field type and utilities for extracting
//! Lean4 representations of Jolt components.

pub mod constants;
pub mod mle_ast;
pub mod util;
pub mod lookups;
pub mod instruction;
pub mod r1cs;
pub mod lean_tests;
pub mod modules;

// Re-export commonly used types
pub use mle_ast::{MleAst, DefaultMleAst};

// Re-export commitment types and their thread-local accessors
pub use mle_ast::{
    AstCommitment, set_pending_commitment_chunks, take_pending_commitment_chunks,
};

