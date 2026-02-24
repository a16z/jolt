//! ZkLean Extractor Library
//!
//! Provides the MleAst symbolic field type and utilities for extracting
//! Lean4 representations of Jolt components.
//!
//! ## Module Structure
//!
//! - `mle_ast`: Core AST types (MleAst, Node, Atom, Edge) and JoltField implementation
//! - `scalar_ops`: Modular arithmetic for BN254 scalar field elements
//! - `ast_bundle`: Serializable IR types for transpilation (AstBundle, AstCommitment)

// Core modules (upstream zklean + our modifications)
pub mod constants;
pub mod mle_ast;
pub mod util;
pub mod lookups;
pub mod instruction;
pub mod r1cs;
pub mod lean_tests;
pub mod modules;

// New modules (100% our code for transpilation)
pub mod scalar_ops;
pub mod ast_bundle;

// Re-export commonly used types from mle_ast
pub use mle_ast::{DefaultMleAst, MleAst};

// Re-export thread-local accessors from mle_ast
pub use mle_ast::{
    set_pending_commitment_chunks, set_pending_point_elements, take_pending_commitment_chunks,
    take_pending_point_elements,
};

// Re-export bundle types (also available via mle_ast for backward compat)
pub use ast_bundle::{
    Assertion, AstBundle, AstCommitment, Constraint, InputKind, InputVar, TargetField,
};

// Re-export scalar ops
pub use scalar_ops::{
    scalar_add_mod, scalar_mul_mod, scalar_neg_mod, scalar_sub_mod, scalar_to_decimal_string,
    BN254_MODULUS, SCALAR_ONE, SCALAR_ZERO,
};


