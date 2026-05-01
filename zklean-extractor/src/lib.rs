//! ZkLean Extractor Library
//!
//! Provides the MleAst symbolic field type for transpiling Jolt verifier
//! operations to external circuit representations.
//!
//! ## Module Structure
//!
//! - `mle_ast`: Core AST types (MleAst, Node, Atom, Edge) and JoltField implementation
//! - `scalar_ops`: Modular arithmetic for BN254 scalar field elements
//! - `ast_bundle`: Serializable IR types for transpilation (AstBundle, AstCommitment)

// Lean extraction modules
pub mod constants;
pub mod instruction;
pub mod lean_tests;
pub mod lookup_table_flags;
pub mod lookups;
pub mod modules;
pub mod r1cs;
pub mod sumchecks;
pub mod util;

// Transpilation modules
pub mod ast_bundle;
pub mod mle_ast;
pub mod scalar_ops;

// Re-export core types
pub use ast_bundle::{Assertion, AstBundle, AstCommitment, TargetField, WitnessType};
pub use mle_ast::{
    set_pending_commitment_chunks, set_pending_point_elements, take_pending_commitment_chunks,
    take_pending_point_elements,
};
pub use mle_ast::{DefaultMleAst, MleAst};
